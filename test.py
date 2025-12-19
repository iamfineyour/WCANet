import torch
import os
import cv2
import numpy as np
import py_sod_metrics
from WCANet import WCANet
import matplotlib.pyplot as plt
from config import opt
from rgbt_dataset import test_dataset
from datetime import datetime
import torch.nn.functional as F
# --------------------- 主测试代码 ---------------------

dataset_path = opt.test_path

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU:', opt.gpu_id)

# 加载模型
model = WCANet()
print('NOW USING: WaveNet')
model.load_state_dict(torch.load('D:/PaperCode/WaveNet/RGBT_dataset/save/Best_mae_test.pth'))
model.cuda()
model.eval()

# 测试数据集
test_datasets = ['VT821','VT1000','VT5000']
all_results = {}  # 存储所有数据集的评估结果

for dataset in test_datasets:
    print(f"\n===== 开始测试 {dataset} =====")
    # 初始化评估指标
    FM = py_sod_metrics.Fmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    MSIOU = py_sod_metrics.MSIoU()
    WFM = py_sod_metrics.WeightedFmeasure()

    sample_gray = dict(with_adaptive=True, with_dynamic=True)
    sample_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=True)
    overall_bin = dict(with_adaptive=False, with_dynamic=False, with_binary=True, sample_based=False)
    FMv2 = py_sod_metrics.FmeasureV2(
        metric_handlers={
            "fm": py_sod_metrics.FmeasureHandler(**sample_gray, beta=0.3),
            "f1": py_sod_metrics.FmeasureHandler(**sample_gray, beta=1),
            "pre": py_sod_metrics.PrecisionHandler(**sample_gray),
            "rec": py_sod_metrics.RecallHandler(**sample_gray),
            "sample_bifm": py_sod_metrics.FmeasureHandler(**sample_bin, beta=0.3),
            "overall_bifm": py_sod_metrics.FmeasureHandler(**overall_bin, beta=0.3),
        }
    )

    mae_sum = 0
    save_path = 'show/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/T/'

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    total_samples = test_loader.size
    print(f"数据集大小: {total_samples} 样本")
    prec_time = datetime.now()

    for i in range(total_samples):
        # 加载数据
        image, gt, depth, name = test_loader.load_data()
        gt = gt.cuda()
        image = image.cuda()
        depth = depth.cuda()

        # 模型推理
        res = model(image, depth)
        res = torch.sigmoid(res[0])

        res = F.interpolate(res, size=gt.shape[2:], mode='bilinear', align_corners=False)

        # 归一化（保持与原代码一致）
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # 计算第一个 MAE（张量层面，无损耗）
        mae_train = torch.sum(torch.abs(res - gt)) / torch.numel(gt)
        mae_sum += mae_train.item()

        # 保存预测结果（尺寸与 GT 一致）
        predict = res.data.cpu().numpy().squeeze()
        # 直接保存原始尺寸，无需强制缩放
        plt.imsave(os.path.join(save_path, name), predict, cmap='gray')

        # --------------------- 评估指标计算（优化版） ---------------------
        # 直接使用张量数据计算指标，避免图像读写和缩放损耗
        pred_np = predict * 255  # 转换为 0-255 范围
        gt_np = gt.data.cpu().numpy().squeeze() * 255  # GT 也转换为 0-255

        # 确保数据类型为 uint8
        pred_np = pred_np.astype(np.uint8)
        gt_np = gt_np.astype(np.uint8)

        # 更新评估指标（直接使用 numpy 数组）
        FM.step(pred=pred_np, gt=gt_np)
        WFM.step(pred=pred_np, gt=gt_np)
        SM.step(pred=pred_np, gt=gt_np)
        EM.step(pred=pred_np, gt=gt_np)
        MAE.step(pred=pred_np, gt=gt_np)
        MSIOU.step(pred=pred_np, gt=gt_np)
        FMv2.step(pred=pred_np, gt=gt_np)

        if (i + 1) % 100 == 0:
            print(f"处理进度: {i + 1}/{total_samples}")

    # 计算测试时间和评估结果
    cur_time = datetime.now()
    avg_mae = mae_sum / total_samples

    # 获取所有评估指标
    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae_result = MAE.get_results()["mae"]
    msiou = MSIOU.get_results()["msiou"]
    fmv2_results = FMv2.get_results()

    # 存储结果
    all_results[dataset] = {
        "MAE": mae_result,
        "Smeasure": sm,
        "adpEm": em["adp"],
        "adpFm": fm["adp"],
        "meanEm": em["curve"].mean(),
        "meanFm": fm["curve"].mean(),
        "wFmeasure": wfm,
        "MSIOU": msiou,
        "maxEm": em["curve"].max(),
        "maxFm": fm["curve"].max(),
        "sample_bifm": fmv2_results["sample_bifm"]["binary"],
        "overall_bifm": fmv2_results["overall_bifm"]["binary"],
    }

    print(f"{dataset} 测试完成，张量层面 MAE: {avg_mae:.6f}, 评估器 MAE: {mae_result:.6f}, 耗时: {cur_time - prec_time}")

# --------------------- 打印汇总结果 ---------------------
print("\n\n===== 所有测试集的汇总评估结果 =====")
for dataset, metrics in all_results.items():
    print(f"\n{dataset}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")

print('Test Done!')
