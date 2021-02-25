# Author: Acer Zhang
# Datetime: 2021/2/25 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle

from paddle.vision.transforms import Compose, Resize, ToTensor
from paddle.vision.models import resnet50
from paddle.vision.datasets import Cifar100

# 导入RIFLE模块
from paddle_rifle.rifle import RIFLE

# 定义数据预处理
transform = Compose([Resize(224),
                     ToTensor()])

# 加载Cifar100数据集
train_data = Cifar100(transform=transform)
test_data = Cifar100(mode="test", transform=transform)

# 加载Resnet50
net = resnet50(True, num_classes=100)
# 获取Resnet50的输出层
fc_layer = net.fc

"""
# 自定义网络输出层获取示例

class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = xxx
        self.layer2 = xxx
        self.输出层 = paddle.nn.Linear(...)
    ...
    
# 加载Net
net = Net()
# 获取输出层
输出层 = net.输出层
"""

model = paddle.Model(network=net,
                     inputs=paddle.static.InputSpec([3, 224, 224], name="ipt"),
                     labels=paddle.static.InputSpec([1], dtype="int64", name="lab"))

# 实例化可视化Callback和RIFLE Callback
vdl = paddle.callbacks.VisualDL("./logB")
rifle_cb = RIFLE(fc_layer, 3, 3)

adam = paddle.optimizer.SGD(parameters=model.parameters())
loss = paddle.nn.loss.CrossEntropyLoss()
acc = paddle.metric.Accuracy((1, 5))
model.prepare(adam, loss, acc)

# 开始训练并传入RIFLE Callback
model.fit(train_data,
          test_data,
          batch_size=128,
          epochs=20,
          log_freq=200,
          callbacks=[vdl, rifle_cb])
