# Author: Acer Zhang
# Datetime: 2021/3/19 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

# Author: Acer Zhang
# Datetime: 2021/2/25
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle
import paddle.nn as nn
from paddle.vision.transforms import Compose, ToTensor
from paddle.vision.datasets import MNIST

# 导入RIFLE模块
from paddle_rifle.rifle import RIFLECallback

transform = Compose([ToTensor()])

train_data = MNIST(transform=transform)
test_data = MNIST(mode="test", transform=transform)


class Net(nn.Layer):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2D(1, 3, 3)
        self.mp = nn.MaxPool2D(2)
        self.conv2 = nn.Conv2D(3, 16, 3)
        self.mp2 = nn.MaxPool2D(2)
        self.fc1 = nn.Linear(400, 100)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.mp(x)
        x = self.conv2(x)
        x = self.mp2(x)
        x = paddle.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


net = Net(num_classes=10)
fc_layer = net.fc2

model = paddle.Model(network=net,
                     inputs=paddle.static.InputSpec([1, 28, 28], name="ipt"),
                     labels=paddle.static.InputSpec([1], dtype="int64", name="lab"))

rifle_cb = RIFLECallback(fc_layer, 1, 3, weight_initializer=paddle.nn.initializer.XavierNormal())

sgd = paddle.optimizer.SGD(parameters=model.parameters())
loss = paddle.nn.loss.CrossEntropyLoss()
acc = paddle.metric.Accuracy((1, 5))
model.prepare(sgd, loss, acc)

# 开始训练并传入RIFLE Callback
model.fit(train_data,
          test_data,
          batch_size=32,
          epochs=20,
          log_freq=100,
          callbacks=[rifle_cb])
