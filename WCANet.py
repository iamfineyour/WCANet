import torch
import torch.nn as nn
from networks.wavemlp import WaveMLP_S
from timm.models.layers import DropPath
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from torch.nn.functional import kl_div
import torch.nn.functional as F
from pytorch_msssim import MS_SSIM
from networks.swinNet import SwinTransformer,SwinNet
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1,output_padding=0, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,output_padding= output_padding, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.inch = in_planes
    def forward(self, x):

        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features=64, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # print('x',x.shape)
        x = self.fc1(x)
        # print('fc',x.shape)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SGS(nn.Module):
    def __init__(self, channels, reduction=4, dilations=[1, 3, 7]):
        super().__init__()
        self.reduced_channels = channels // reduction

        # 多尺度空洞卷积（避免上/下采样）
        self.scales = nn.ModuleList()
        for dilation in dilations:
            self.scales.append(nn.Sequential(
                nn.Conv2d(channels, self.reduced_channels, 3,
                          padding=dilation, dilation=dilation),
                nn.GroupNorm(4, self.reduced_channels),
                nn.GELU()
            ))

        # 精简门控生成器
        self.gate1 = nn.Sequential(
            nn.Conv2d(3 * self.reduced_channels, channels, 1),  # 通道压缩
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, 2, 1),  # 直接输出两个模态的权重
            nn.Sigmoid()  # 使用Sigmoid约束到[0,1]
        )

        # 特征整合层
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(4, channels),
            nn.GELU()
        )
        self.SA = SALayer()

    def forward(self, rgb, thermal):
        # 提取多感受野特征（保持分辨率）
        scale_feats = [scale(rgb) for scale in self.scales]
        multi_feat = torch.cat(scale_feats, dim=1)

        # 生成门控信号
        gates = self.gate1(multi_feat)
        gate_rgb, gate_thermal = gates.chunk(2, dim=1)

        # 安全的门控融合
        fused = gate_rgb * rgb + gate_thermal * thermal
        fused = self.fusion_conv(fused)
        fused = self.SA(fused)
        return fused


class MVS(nn.Module):
    """
    通道级均值-方差标准化 (Channel-Wise Mean-Variance Standardization)对每个通道的每个空间位置，计算其在通道内的均值和方差,更适用于保留局部结构特征
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
            x: 输入特征图 [B, C, H, W],归一化后的特征图 [B, C, H, W]
        """
        # 计算通道维度上的均值和方差
        # 保持维度以便广播
        mean = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        var = torch.var(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # 归一化
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        return normalized


class SimFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.cw_mvn = MVS()
        self.cos_sim = nn.CosineSimilarity(dim=1)

        # 最终融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, thermal):
        # 归一化

        # 计算相似度
        cos_sim = self.cos_sim(rgb, thermal).unsqueeze(1)
        M = (cos_sim + 1) / 2
        # 共识路径
        f1 = M * rgb + M * thermal

        return f1


class SEW(nn.Module):
    def __init__(self, inchannel, outchannel):
        super().__init__()
        self.DWT = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
        self.IWT = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        # Sigmoid激活：生成0~1的通道权重
        self.sigmoid = nn.Sigmoid()
        self.conv1x1 = nn.Conv2d(in_channels=inchannel, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(1)
        # 通道对齐
        self.conv1 = BasicConv2d(outchannel, outchannel)
        self.conv2 = BasicConv2d(inchannel, outchannel)
        self.conv3 = BasicConv2d(outchannel, outchannel)
        self.conv4 = BasicConv2d(outchannel, outchannel)
        self.SGS = SGS(inchannel)  # 替换为多尺度门控

        self.gate2 = nn.Sequential(
            nn.Conv2d(outchannel * 2, outchannel, 3, padding=1),  # 特征压缩
            nn.GroupNorm(4, outchannel),
            nn.GELU(),
            nn.Conv2d(outchannel, 2, 1),  # 输出2个权重图
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        self.Sim = SimFusion(outchannel)

    def forward(self, x, y):
        y = self.conv2(y)
        xl, xh = self.DWT(x)
        yl, yh = self.DWT(y)
        f = self.SGS(x, y)
        xl = self.conv1(xl)
        yl = self.conv1(yl)

        low = self.Sim(xl, yl)

        x_spatial = self.conv1x1(low)  # 压缩通道至1，聚焦空间分布，shape=[batch, 1, H, W]
        # 步骤2：批归一化稳定训练
        x_spatial = self.bn(x_spatial)
        # 步骤3：Sigmoid生成空间注意力权重ω₂（文档中公式5：ω₂=R(F(X^S))）
        omega2 = self.sigmoid(x_spatial)  # 权重范围[0,1]，shape=[batch, 1, H, W]
        # 使用双线性插值，size 设为 f 的 H 和 W（f.shape[2:] 即 [H_high, W_high]）
        omega2 = F.interpolate(
            omega2,
            size=f.shape[2:],  # 对齐 f 的空间维度
            mode='bilinear',  # 适合图像的插值方式
            align_corners=False  # 避免边缘伪影
        )  # 此时 omega2 形状为 [B, 1, H_high, W_high]，与 f 匹配
        f = f * omega2  # 广播机制，逐空间位置加权

        xm = self.IWT((low, xh))
        ym = self.IWT((low, yh))

        out = self.conv3(xm+ym)
        # 特征融合
        two_input = torch.cat([out, self.conv4(f)], dim=1)
        weights = self.gate2(two_input)
        w_out, w_f = weights.chunk(2, dim=1)

        fused = out * (1 + w_out) + f * (1 + w_f)
        return fused


class SALayer(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),  # 增加非线性
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        std_pool = torch.std(x, dim=1, keepdim=True)  # 新增标准差特征
        feat = torch.cat([max_pool, std_pool], dim=1)
        weights = self.conv(feat)
        return x * weights


class PFGF(nn.Module):
    def __init__(self, channel_1, channel_2, channel_3, dilation_1=2, dilation_2=3):
        super().__init__()
        self.conv1 = BasicConv2d(channel_1, channel_2, 3, padding=1)
        self.conv0 = BasicConv2d(channel_1, channel_3, 3, padding=1)
        self.conv1_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)
        self.conv2 = BasicConv2d(channel_2, channel_2, 3, padding=1)
        self.conv2_Dila = BasicConv2d(channel_2, channel_2, 3, padding=dilation_1, dilation=dilation_1)

        self.conv31 = BasicConv2d(channel_2, channel_3, 3, padding=dilation_2, dilation=dilation_2)
        self.conv_fuse = BasicConv2d(channel_2 * 2, channel_3, 3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.conv_last = TransBasicConv2d(channel_3, channel_3, kernel_size=2, stride=2,
                                          padding=0, dilation=1, bias=False)

        self.SA = SALayer()

    def forward(self, x):
        x0 = self.conv0(x)
        x4 = self.SA(x0)

        x1 = self.conv1(x)
        x1_dila = self.conv1_Dila(x1)
        x2 = self.conv2(x1)
        x2_dila = self.conv2_Dila(x2)

        x3 = self.conv31(x2)
        x_fuse = self.conv_fuse(torch.cat((x1_dila, x2_dila), 1))
        x_fuse = (x_fuse + x3)*x4+x0
        x_fuse = self.drop(x_fuse)
        # print()
        x_fuse = self.conv_last(x_fuse)
        return x_fuse

class Edge_Aware(nn.Module):
    def __init__(self, ):
        super(Edge_Aware, self).__init__()
        self.conv1 = TransBasicConv2d(512, 64,kernel_size=4,stride=8,padding=0,dilation=2,output_padding=1)
        self.conv2 = TransBasicConv2d(320, 64,kernel_size=2,stride=4,padding=0,dilation=2,output_padding=1)
        self.conv3 = TransBasicConv2d(128, 64,kernel_size=2,stride=2,padding=1,dilation=2,output_padding=1)
        self.pos_embed = BasicConv2d(64, 64 )
        self.pos_embed3 = BasicConv2d(64, 64)
        self.conv31 = nn.Conv2d(64,1, kernel_size=1)
        self.conv512_64 = TransBasicConv2d(512,64)
        self.conv320_64 = TransBasicConv2d(320, 64)
        self.conv128_64 = TransBasicConv2d(128, 64)
        self.up = nn.Upsample(56)
        self.up2 = nn.Upsample(384)
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.BatchNorm2d(64)
        self.drop_path = DropPath(0.3)
        self.maxpool =nn.AdaptiveMaxPool2d(1)
        # self.qkv = nn.Linear(64, 64 * 3, bias=False)
        self.num_heads = 8
        self.mlp1 = Mlp(in_features=64, out_features=64)
        self.mlp2 = Mlp(in_features=64, out_features=64)
        self.mlp3 = Mlp(in_features=64, out_features=64)
    def forward(self, x, y, z, v):


        # v = self.conv1(v)
        # z = self.conv2(z)
        # y = self.conv3(y)
        # print('v',v)
        v = self.up(self.conv512_64(v))
        z = self.up(self.conv320_64(z))
        y = self.up(self.conv128_64(y))
        x = self.up(x)

        x_max = self.maxpool(x)
        # print('x_max',x_max.shape)
        b,_,_,_ = x_max.shape
        x_max = x_max.reshape(b, -1)
        x_y = self.mlp1(x_max)
        # print('s',x_y.shape)
        x_z = self.mlp2(x_max)
        x_v = self.mlp3(x_max)

        x_y = x_y.reshape(b,64,1,1)
        x_z = x_z.reshape(b, 64, 1, 1)
        x_v = x_v.reshape(b, 64, 1, 1)
        x_y = torch.mul(x_y, y)
        x_z = torch.mul(x_z, z)
        x_v = torch.mul(x_v, v)


        # x_mix_1 = torch.cat((x_y,x_z,x_v),dim=1)
        x_mix_1 = x_y+ x_z+ x_v
        # print('sd',x_mix_1.shape)
        x_mix_1 = self.norm2(x_mix_1)
        # print('x_mix_1',x_mix_1.shape)
        x_mix_1 = self.pos_embed3(x_mix_1)
        x_mix = self.drop_path(x_mix_1)
        x_mix = x_mix_1 + self. pos_embed3(x_mix)
        x_mix = self.up2(self.conv31(x_mix))
        return x_mix

class Mutual_info_reg(nn.Module):
    def __init__(self, input_channels=64, channels=64, latent_size=6):
        super(Mutual_info_reg, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)
    def forward(self, rgb_feat, depth_feat):

        # print('rgb_feat',rgb_feat.shape)
        # print('depth_feat', depth_feat.shape)
        rgb_feat = self.soft(rgb_feat)
        depth_feat = self.soft(depth_feat)
        #
        # print('rgb_feat',rgb_feat.shape)
        # print('depth_feat', depth_feat.shape)
        return kl_div(rgb_feat.log(), depth_feat)

class WCANet(nn.Module):
    def __init__(self, channel=32):
        super(WCANet, self).__init__()
        self.encoderR = WaveMLP_S()
        # Lateral layers
        self.SEW1 = SEW(64,64)
        self.SEW2 = SEW(128,128)
        self.SEW3 = SEW(320,320)
        self.SEW4 = SEW(512,512)

        self.conv512_64 = BasicConv2d(512, 64)
        self.conv320_64 = BasicConv2d(320, 64)
        self.conv128_64 = BasicConv2d(128, 64)
        self.sigmoid = nn.Sigmoid()
        self.S4 = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.S3 = nn.Conv2d(320, 1, 3, stride=1, padding=1)
        self.S2 = nn.Conv2d(128, 1, 3, stride=1, padding=1)
        self.S1 = nn.Conv2d(64, 1, 3, stride=1, padding=1)
        self.up1 = nn.Upsample(384)
        self.up2 = nn.Upsample(384)
        self.up3 = nn.Upsample(384)
        self.up_loss = nn.Upsample(92)
        # Mutual_info_reg1
        self.mi_level1 = Mutual_info_reg(64, 64, 6)
        self.mi_level2 = Mutual_info_reg(64, 64, 6)
        self.mi_level3 = Mutual_info_reg(64, 64, 6)
        self.mi_level4 = Mutual_info_reg(64, 64, 6)

        self.edge = Edge_Aware()
        self.PATM4 = PFGF(512, 512, 512, 2, 3)
        self.PATM3 = PFGF(832, 512, 320, 2, 3)
        self.PATM2 = PFGF(448, 256, 128, 4, 5)
        self.PATM1 = PFGF(192, 128, 64, 4, 5)

    def forward(self, x_rgb,x_thermal):
        x0,x1,x2,x3 = self.encoderR(x_rgb)
        y0, y1, y2, y3 = self.encoderR(x_thermal)

        x2_ACCoM = self.SEW1(x0, y0)
        x3_ACCoM = self.SEW2(x1, y1)
        x4_ACCoM = self.SEW3(x2, y2)
        x5_ACCoM = self.SEW4(x3, y3)

        edge = self.edge(x2_ACCoM, x3_ACCoM, x4_ACCoM, x5_ACCoM)

        mer_cros4 = self.PATM4(x5_ACCoM)
        m4 = torch.cat((mer_cros4,x4_ACCoM),dim=1)
        mer_cros3 = self.PATM3(m4)
        m3 = torch.cat((mer_cros3, x3_ACCoM), dim=1)
        mer_cros2 = self.PATM2(m3)
        m2 = torch.cat((mer_cros2, x2_ACCoM), dim=1)
        mer_cros1 = self.PATM1(m2)

        s1 = self.up1(self.S1(mer_cros1))
        s2 = self.up2(self.S2(mer_cros2))
        s3 = self.up3(self.S3(mer_cros3))
        s4 = self.up3(self.S4(mer_cros4))

        x_loss0 = x0
        y_loss0 = y0
        x_loss1 = self.up_loss(self.conv128_64(x1))
        y_loss1 = self.up_loss(self.conv128_64(y1))
        x_loss2 = self.up_loss(self.conv320_64(x2))
        y_loss2 = self.up_loss(self.conv320_64(y2))
        x_loss3 = self.up_loss(self.conv512_64(x3))
        y_loss3 = self.up_loss(self.conv512_64(y3))

        lat_loss0 = self.mi_level1(x_loss0, y_loss0)
        lat_loss1 = self.mi_level2(x_loss1, y_loss1)
        lat_loss2 = self.mi_level3(x_loss2, y_loss2)
        lat_loss3 = self.mi_level4(x_loss3, y_loss3)
        lat_loss = lat_loss0 + lat_loss1 + lat_loss2 + lat_loss3
        return s1, s2, s3, s4, self.sigmoid(s1), self.sigmoid(s2), self.sigmoid(s3), self.sigmoid(s4),edge,lat_loss
    def load_pre(self, path):
        """
        从指定路径加载预训练权重。
        参数:
        path (str): 预训练权重文件的路径。
        """
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=False)
        print("预训练权重已成功加载。")
if __name__=='__main__':
    image = torch.randn(1, 3, 384, 384).cuda(0)
    ndsm = torch.randn(1, 64, 56, 56)
    ndsm1 = torch.randn(1, 128, 28, 28)
    ndsm2 = torch.randn(1, 320, 14, 14)
    ndsm3 = torch.randn(1, 512, 7, 7)

    net = WCANet().cuda()

