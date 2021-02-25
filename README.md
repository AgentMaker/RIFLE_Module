# 可用于PaddlePaddle的RIFLE优化策略模块

## 简介
RIFLE优化策略会在训练中随机初始化输出层，从而让模型更关注深层网络的更新，从而可以在图像分类等任务中取得较优效果。  
本项目则为可用于PaddlePaddle的RIFLE优化策略封装版，支持普通API与高阶API，并且只需向训练代码中插入一行代码即可使用RIFLE策略。  
原作论文以及本项目性能等相关详见README底部。

当前模块完成进度：
- [X] 输出层基于Linear的分类任务
+ [X] 输出层基于Linear的语义分割任务
+ [ ] 目标检测

## 使用方法
### 安装
`pip install paddle-rifle`  
若上方命令安装失败可尝试使用下方命令：  
`pip install paddle-rifle -i https://pypi.tuna.tsinghua.edu.cn/simple` 
### 在组网中加入RIFLE
#### 方案一、使用飞桨高层API添加RIFLE策略 - 完整代码详见`demo.py`
```
from rifle import RIFLE

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
    rifle_callback = RIFLE(layer=out_layer, re_init_epoch=5)
    ...
    # 使用Hapi进行训练
    model.fit(..., callbacks=[rifle_callback])
```
#### 方案二、基于飞桨基础API添加RIFLE策略
```
from rifle import RIFLE

class YourModel:
        def __init__(self):
            super(LeNet, self).__init__():
            ...
            # 定义输出层
            self.out_layer = paddle.nn.Linear(...)
            ...
        ...
...
# 实例化YourModel
model = YourModel()
...
# 开始训练
for current_epoch in range(EPOCH_NUM):
    for data in data_loader():
        ...
        # 训练部分代码
        ...
        
    # 加入RIFLE策略
    rifle(layer=model.out_layer, current_epoch=current_epoch, re_init_epoch=5)
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
