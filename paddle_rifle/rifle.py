# Author: Acer Zhang
# Datetime: 2021/2/23 
# Copyright belongs to the author.
# Please indicate the source for reprinting.
import copy

import paddle


class RIFLE:
    """
    RIFLE实现
    :param layers: 需要重置的Layer 或 Layer列表
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
    # 实例化RIFLE策略
    rifle = RIFLE(layer=model.out_layer, re_init_epoch=5)
    # 开始训练
    for current_epoch in range(EPOCH_NUM):
        for data in data_loader():
            ...
        # 加入RIFLE策略
        rifle.apply(current_epoch=current_epoch)
    """

    def __init__(self, layers, re_init_epoch: int = 5, max_re_num: int = 3):
        if not isinstance(layers, list):
            layers = [layers]
        self.layers = layers
        self.re_init_epoch = re_init_epoch
        self.max_re_num = max_re_num
        self.CACHE_PARAMS = dict()

    def apply(self, current_epoch: int):
        """
        应用RIFLE
        :param current_epoch: 当前遍历过的EPOCH数量
        :return:
        """
        if current_epoch % self.re_init_epoch == 0 and (current_epoch // self.re_init_epoch) <= self.max_re_num:
            print_str = f"Initialization successful, {len(self.layers)} layers will apply RIFLE"

            if current_epoch == 0:
                for layer_id, layer in enumerate(self.layers):
                    self.CACHE_PARAMS[layer_id] = copy.deepcopy(layer.parameters())
            else:
                for layer_id, layer in enumerate(self.layers):
                    for param in self.CACHE_PARAMS[layer_id]:
                        if ".w_" in param.name:
                            layer.weight = layer.create_parameter(shape=layer.weight.shape,
                                                                  attr=None,
                                                                  dtype=layer.weight.dtype,
                                                                  is_bias=False)
                        elif ".b_" in param.name:
                            layer.bias = layer.create_parameter(shape=layer.bias.shape,
                                                                attr=None,
                                                                dtype=layer.bias.dtype,
                                                                is_bias=True)
                print_str = f"RIFLE: The output layer has been reset in the {current_epoch} epoch!"

            print(f"\033[0;37;41m{print_str}\033[0m")


class RIFLECallback(paddle.callbacks.Callback):
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

    def __init__(self, layers, re_init_epoch: int = 5, max_re_num: int = 3):
        """
        RIFLE的CallBack实现
        :param layers: 需要进行RIFLE的输出层
        :param re_init_epoch: 经历多少EPOCH后重新初始化输出层
        :param max_re_num: Layer最大重置次数
        """
        super(RIFLECallback, self).__init__()
        self.re_init_epoch = re_init_epoch
        self.layer = layers
        self.max_re_num = max_re_num
        self._rifle = RIFLE(layers=layers,
                            re_init_epoch=re_init_epoch,
                            max_re_num=max_re_num)

    def on_epoch_begin(self, epoch, logs=None):
        self._rifle.apply(epoch)
