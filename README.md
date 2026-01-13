# WCANet for RGB-T SOD
This is an official implementation for **“Wave-Driven Structure-Semantic Co-Alignment Network for RGB-T Salient Object Detection”**

## Requirements
Environmental Setups: Python ≥ 3.7, Pytorch ≥ 1.6.0, Torchvision ≥0.7.0, pytorch_wavelets.

## Preparation
Both our training sets and test sets are based on public datasets, which can be searched for and downloaded online.You can also obtain the data we used here: Baidu Netdisk link:https://pan.baidu.com/s/1RTkraZBNOS3mDt-KmLNIqw?pwd=mevr pin: mevr
Download the pre-trained wavemlp-s from [wavemlp.](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/wavemlp_pytorch)  
You can load the SwinNet model for knowledge distillation training, please refer to the specific configuration of [SwinNet.](https://github.com/liuzywen/SwinNet)  
Then, you should run test.py or train.py.
## Evaluate tools
You can use this toolbox to calculate performance metrics [COD Toolbox.](https://github.com/DengPingFan/CODToolbox)
## Results of WCANet
You can access our salient prediction maps via the Baidu Netdisk link, including those from RGBT datasets (VT821, VT1000, VT5000, VI-RGBT1500) and four RGBD datasets.  
Link: https://pan.baidu.com/s/15tTWCn8uUS_K8ftJOKpyTg?pwd=5myx pin: 5myx.


<img width="867" height="451" alt="image" src="https://github.com/user-attachments/assets/d8ebb763-f145-4238-90f9-6f82a167e06c" /><br>  
<img width="415" height="241" alt="image" src="https://github.com/user-attachments/assets/9d1ac424-9769-4433-95d5-6efd3dfa53db" /><br>
<img width="414" height="415" alt="image" src="https://github.com/user-attachments/assets/f57fd32c-43fe-44e9-bc34-c9c2ee1f8e3e" /><br>

## Pretraining Model
You can obtain our pre-trained model from the link: https://pan.baidu.com/s/1O4Wv-7MODvP9u0_QgPW3TQ?pwd=agv6 pin: agv6.

## Contact
If you have any questions or discussions, please feel free to contact me.(cy_bai@smail.sut.edu.cn)
