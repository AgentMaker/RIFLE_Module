# Author: Acer Zhang
# Datetime: 2021/2/23 
# Copyright belongs to the author.
# Please indicate the source for reprinting.

import paddle


class RIFLE:
    """
    RIFLE实现

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

    def __init__(self,
                 layers,
                 re_init_epoch: int = 5,
                 max_re_num: int = 3,
                 weight_initializer=None):
        """
        :param layers: 需要重置的Layer 或 Layer列表
        :type layers: (paddle.nn.Layer|list)
        :param re_init_epoch: 经历多少EPOCH后重新初始化输出层
        :param max_re_num: Layer最大重置次数
        :param weight_initializer: 权重默认初始化方案（若为None则为原始权重，可为paddle.nn.initializer.XavierNormal()）
        """
        if not isinstance(layers, list):
            layers = [layers]
        self.layers = layers
        self.re_init_epoch = re_init_epoch
        self.max_re_num = max_re_num
        self.weight_initializer = weight_initializer

        self.CACHE_PARAMS = dict()

    def apply(self, current_epoch: int):
        """
        应用RIFLE
        :param current_epoch: 当前遍历过的EPOCH数量
        """
        if current_epoch == 0:
            print(f"\033[0;37;41mInitialization successful, {len(self.layers)} group layers will apply RIFLE\033[0m")
        elif current_epoch % self.re_init_epoch == 0 and (current_epoch // self.re_init_epoch) <= self.max_re_num:
            for layer_id, layer in enumerate(self.layers):
                if self.weight_initializer is None:
                    layer.parameters().clear()
                else:
                    for param_id, param in enumerate(layer.parameters()):
                        if ".w_" in param.name:
                            layer.weight = layer.create_parameter(shape=layer.weight.shape,
                                                                  attr=None,
                                                                  dtype=layer.weight.dtype,
                                                                  is_bias=False,
                                                                  default_initializer=self.weight_initializer)
                        elif ".b_" in param.name:
                            layer.bias = layer.create_parameter(shape=layer.bias.shape,
                                                                attr=None,
                                                                dtype=layer.bias.dtype,
                                                                is_bias=True)

            print(f"\033[0;37;41mRIFLE:  layer has been reset in the {current_epoch} epoch!\033[0m")


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

    def __init__(self,
                 layers,
                 re_init_epoch: int = 5,
                 max_re_num: int = 3,
                 weight_initializer=None):
        """
        RIFLE的CallBack实现
        :param layers: 需要进行RIFLE的输出层
        :type layers: (paddle.nn.Layer|list)
        :param re_init_epoch: 经历多少EPOCH后重新初始化输出层
        :param max_re_num: Layer最大重置次数
        :param weight_initializer: 权重默认初始化方案（若为None则为原始权重，可为paddle.nn.initializer.XavierNormal()）
        """
        super(RIFLECallback, self).__init__()
        self.re_init_epoch = re_init_epoch
        self.layer = layers
        self.max_re_num = max_re_num
        self._rifle = RIFLE(layers=layers,
                            re_init_epoch=re_init_epoch,
                            max_re_num=max_re_num,
                            weight_initializer=weight_initializer)

    def on_epoch_begin(self, epoch, logs=None):
        self._rifle.apply(epoch)
