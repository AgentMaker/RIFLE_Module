# Author: Acer Zhang
# Datetime: 2021/2/23 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import paddle


def rifle(layer, current_epoch, re_init_epoch=5, max_re_num=3):
    """
    RIFLE实现
    :param layer: 需要重置的Layer
    :param current_epoch: 当前的训练轮数
    :param re_init_epoch: 经历多少EPOCH后重新初始化输出层
    :param max_re_num: Layer最大重置次数

    Example:
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
    # 开始训练
    for current_epoch in range(EPOCH_NUM):
        for data in data_loader():
            ...
        # 加入RIFLE策略
        rifle(layer=model.out_layer, current_epoch=current_epoch, re_init_epoch=5)
    """
    if current_epoch % re_init_epoch == 0 and (current_epoch // re_init_epoch) <= max_re_num:
        param_shape = layer.weight.shape
        layer.weight = layer.create_parameter(shape=param_shape,
                                              attr=layer._weight_attr,
                                              dtype=layer._dtype,
                                              is_bias=False)
        layer.bias = layer.create_parameter(shape=[param_shape[1]],
                                            attr=layer._bias_attr,
                                            dtype=layer._dtype,
                                            is_bias=True)
        print(f"RIFLE: The output layer has been reset in the {current_epoch} epoch!")


class RIFLE(paddle.callbacks.Callback):
    """
    RIFLE 在飞桨Hapi中的Callback实现

    Example:
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

    """

    def __init__(self, layer, re_init_epoch=5, max_re_num=3):
        """
        RIFLE的CallBack实现
        :param layer: 需要进行RIFLE的输出层
        :param re_init_epoch: 经历多少EPOCH后重新初始化输出层
        :param max_re_num: Layer最大重置次数
        """
        super(RIFLE, self).__init__()
        self.re_init_epoch = re_init_epoch
        self.layer = layer
        self.max_re_num = max_re_num

    def on_epoch_begin(self, epoch, logs=None):
        rifle(self.layer, epoch, self.re_init_epoch, self.max_re_num)
