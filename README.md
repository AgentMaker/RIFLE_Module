# 可用于PaddlePaddle的RIFLE优化策略模块
![GitHub forks](https://img.shields.io/github/forks/GT-ZhangAcer/RIFLE_Module)
![GitHub Repo stars](https://img.shields.io/github/stars/GT-ZhangAcer/RIFLE_Module)
[![User](https://static.pepy.tech/personalized-badge/paddle-rifle?period=total&units=international_system&left_color=grey&right_color=orange&left_text=User)](https://pepy.tech/project/paddle-rifle)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/GT-ZhangAcer/RIFLE_Module?include_prereleases)
![GitHub](https://img.shields.io/github/license/GT-ZhangAcer/RIFLE_Module)
[![Upload Python Package](https://github.com/GT-ZhangAcer/RIFLE_Module/actions/workflows/python-publish.yml/badge.svg)](https://github.com/GT-ZhangAcer/RIFLE_Module/actions/workflows/python-publish.yml)
## 简介
RIFLE优化策略会在训练中随机初始化输出层，让模型更关注深层网络的更新，从而可以在图像分类等任务中取得较优效果。  
本项目则为可用于PaddlePaddle的RIFLE优化策略封装版，支持普通API与高阶API，并且只需向训练代码中插入一行代码即可使用RIFLE策略。  
原作论文以及本项目性能等相关详见README底部。

当前模块完成进度：
- [X] 分类任务
+ [X] 语义分割任务
- [X] 目标检测 - Beta
+ [ ] 其它类型任务和输出层

## New!版本更新   
V0.2 适配非Linear输出层的网络结构，并支持多层使用RIFLE策略！
## 使用方法
### 安装
`pip install paddle-rifle`  
若上方命令安装失败可尝试使用下方命令：  
`pip install paddle-rifle -i https://pypi.tuna.tsinghua.edu.cn/simple` 

### Paddle-RIFLE API
#### Callback API
```
class RIFLECallback(layers, re_init_epoch, max_re_num)

Callback API 适用于PaddlePaddle 高阶API
:param layers: 需要进行RIFLE的Layer或需要RIFLE的Layers列表
:param re_init_epoch: 经历多少EPOCH后重新初始化输出层
:param max_re_num: Layer最大重置次数
:param weight_initializer: 权重默认初始化方案（:param weight_initializer: 权重默认初始化方案（若为None则为原始权重，可为paddle.nn.initializer.XavierNormal()））
```
#### 常规组网API
```
class RIFLE(layers, re_init_epoch: int = 5, max_re_num: int = 3)

常规组网API 适用于PaddlePaddle常规训练方式
:param layers: 需要重置的Layer 或 Layer列表
:param re_init_epoch: 经历多少EPOCH后重新初始化输出层
:param max_re_num: Layer最大重置次数
:param weight_initializer: 权重默认初始化方案（:param weight_initializer: 权重默认初始化方案（若为None则为原始权重，可为paddle.nn.initializer.XavierNormal()））
```
### 在组网中加入RIFLE
#### 方案一、使用飞桨高层API添加RIFLE策略 - 完整代码详见`demo.py`
```
from paddle_rifle.rifle import RIFLECallback

class YourModel:
        def __init__(self):
            super(LeNet, self).__init__():
            ...
            # 定义输出层
            self.out_layer = paddle.nn.Linear(...)
        ...
...
# 实例化YourModel
model = paddle.Model(YourModel())
out_layer = model.out_layer
rifle_callback = RIFLECallback(layers=out_layer, re_init_epoch=5)
...
# 使用Hapi进行训练
model.fit(..., callbacks=[rifle_callback])
```
#### 方案二、基于飞桨基础API添加RIFLE策略
```
from paddle_rifle.rifle import RIFLE

class YourModel:
    def __init__(self):
        super(LeNet, self).__init__():
        ...
        # 定义输出层
        self.out_layer = paddle.nn.Linear(...)
    ...
...
# 实例化YourModel
model = YourModel()
...
# 实例化RIFLE策略
rifle = RIFLE(layers=model.out_layer, re_init_epoch=5)
# 开始训练
for current_epoch in range(EPOCH_NUM):
    for data in data_loader():
        ...
    # 加入RIFLE策略
    rifle.apply(current_epoch=current_epoch)
```

## 关于
### 本项目的相关指标
使用PaddlePaddle默认划分的CIFAR100交叉验证集中Acc Top1的表现：
<img src="https://ai-studio-static-online.cdn.bcebos.com/6f3dbf89d6f449858d48777a387844b01cb18b8993794912a4653de90b19f927"></img>  

### 相关RIFLE代码参考链接
[https://github.com/haozhe-an/RIFLE-Paddle-Implementation](https://github.com/haozhe-an/RIFLE-Paddle-Implementation)

### 原作者论文信息
> [ICML'20] Xingjian Li*, Haoyi Xiong*, Haozhe An, Dejing Dou, and Cheng-Zhong Xu. RIFLE: Backpropagation in Depth for Deep Transfer Learning through Re-Initializing the Fully-connected LayEr. International Conference on Machine Learning (ICML’20), Vienna, Austria, 2020.

####  Abstract

>Fine-tuning the deep convolution neural network(CNN) using a pre–trained model helps transfer knowledge learned from larger datasets to the target task. While the accuracy could be largely improved even when the training dataset is small, the transfer learning outcome is usually constrained by the pre-trained model with close CNN weights (Liu et al., 2019), as the backpropagation here brings smaller updates to deeper CNN layers. In this work, we propose RIFLE– a simple yet effective strategy that deepens backpropagation in transfer learning settings, through periodically Re-Initializing the Fully connected LayEr with random scratch during the fine-tuning procedure. RIFLE brings meaningful updates to the weights of deep CNN layers and improves low-level feature learning, while the effects of randomization can be easily converged throughout the overall learning procedure. The experiments show that the use of RIFLE significantly improves deep transfer learning accuracy on a wide range of datasets, outperforming known tricks for the similar purpose, such as Dropout, DropConnect, Stochastic Depth, Disturb Label and Cyclic Learning Rate, under the same settings with 0.5%–2% higher testing accuracy. Empirical cases and ablation studies further indicate RIFLE brings meaningful updates to deep CNN layers with accuracy improved.
