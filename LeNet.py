import torch
import torch.nn as nn
import torch.nn.functional as F
from quantnn import *

class LeNet(nn.Module):
    """
    The LeNet-5 module for MINIST dataset.
    """
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = QuantConv2d(1, 6, 5, 1, 2)
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.conv2 = QuantConv2d(6, 16, 5)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.fc1 = QuantLinear(16*5*5, 120)
        self.fc2 = QuantLinear(120, 84)
        self.fc3 = QuantLinear(84, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.maxpool1(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxpool2(out)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out

    def quantize(self, quant_sheme='asymmetric', num_bits=8):
        self.conv1.quantize(quant_sheme, num_bits)
        self.conv2.quantize(quant_sheme, num_bits)
        self.fc1.quantize(quant_sheme, num_bits)
        self.fc2.quantize(quant_sheme, num_bits)
        self.fc3.quantize(quant_sheme, num_bits)


if __name__ == '__main__':
    net = LeNet(10)
    print(net)