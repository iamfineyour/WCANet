import argparse
parser = argparse.ArgumentParser()
# change dir according to your own dataset dirs
# train/val
parser.add_argument('--epoch', type=int, default=80, help='epoch number')
parser.add_argument('--lr', type=float, default=3e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')#0.1
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='None', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
'''
parser.add_argument('--lr_train_root', type=str, default='/root/RGBT_dataset0/train/', help='the train images root')
parser.add_argument('--lr_val_root', type=str, default='/root/RGBT_dataset0/val/', help='the val images root')
parser.add_argument('--save_path', type=str, default='/root/RGBT_dataset0/save/', help='the path to save models and logs')
'''
parser.add_argument('--lr_train_root', type=str, default='D:/PaperCode/WaveNet/RGBT_dataset/train/', help='the train images root')
parser.add_argument('--lr_val_root', type=str, default='D:/PaperCode/WaveNet/RGBT_dataset/val/', help='the val images root')
parser.add_argument('--save_path', type=str, default='D:/PaperCode/WaveNet/RGBT_dataset/save/', help='the path to save models and logs')

# test(predict)
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--test_path',type=str,default='D:/PaperCode/WaveNet/RGBT_dataset/test/',help='test dataset path')
opt = parser.parse_args()
