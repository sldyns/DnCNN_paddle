import paddle
import paddle.nn as nn
import math
import numpy as np
from paddle import ParamAttr


class DnCNN(nn.Layer):
    def __init__(self, channels, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2D(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias_attr=False, weight_attr=nn.initializer.KaimingNormal()))
        # layers.append(nn.Conv2D(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
        #                         bias_attr=False))

        layers.append(nn.ReLU())
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias_attr=False, weight_attr=nn.initializer.KaimingNormal()))

            # layers.append(nn.Conv2D(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
            #               bias_attr=False))

            layers.append(nn.BatchNorm2D(features, weight_attr=ParamAttr(initializer=nn.initializer.Constant(value=1.)),
                                  bias_attr=ParamAttr(initializer=nn.initializer.Constant(value=0.))))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2D(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias_attr=False, weight_attr=nn.initializer.KaimingNormal()))
        # layers.append(nn.Conv2D(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias_attr=False))

        self.dncnn = paddle.nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out
