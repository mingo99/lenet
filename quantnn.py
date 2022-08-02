import torch
import torch.nn as nn
import torch.nn.functional as F
from quant import Quant

class QuantConv2d(nn.Conv2d):
    """
    nn.Conv2d with quantization function.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv2d, self).__init__(in_channels, out_channels, 
            kernel_size, stride, padding, dilation, groups, bias)
        self.quant_flag = False
        self.scale = None
        self.zero_point = None
        self.quant = None
    
    def quantize(self, quant_scheme='asymmetric', num_bits=8):
        """
        Method for quantize the weights.
        """
        self.quant = Quant(quant_scheme, num_bits)
        self.weight.data, self.scale, self.zero_point = self.quant.quantize(self.weight.data)
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            weight = self.quant.dequantize(self.weight, self.scale, self.zero_point)
            return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class QuantLinear(nn.Linear):
    """
    nn.Linear with quantization function. 
    """
    def __init__(self, in_features, out_features, bias=True):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_flag = False
        self.scale = None
        self.zero_point = None
        self.quant = None
    
    def quantize(self, quant_scheme='asymmetric', num_bits=8):
        """
        Method for quantize the weights.
        """
        self.quant = Quant(quant_scheme, num_bits)
        self.weight.data, self.scale, self.zero_point = self.quant.quantize(self.weight.data)
        self.quant_flag = True

    def forward(self, x):
        if self.quant_flag == True:
            weight = self.quant.dequantize(self.weight, self.scale, self.zero_point)
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)