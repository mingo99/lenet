#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description: The class for tensor quantization.
@Date:     2022/07/29 10:17:37
@Author:   Mingo
@version:  1.0
'''

class Quant():
    """
    Quantization operation for torch tensor.
    Support symmetric and asymmetric linear quantization.

    Args:
        x(float): input float32
        scheme(str): quantization scheme, including `asymmetric`, `symmetric_signed` and `symmetric_unsigned`
        num_bits(int): the bit width of int
    """
    def __init__(self, scheme='asymmetric', num_bits=8):
        self.scheme = scheme
        self.num_bits = num_bits
        self.q_x = None
        self.s = None
        self.z = None

    def quantize(self, x):
        """
        Linear quantization method, according to param `scheme` convert float32 to int.
        """
        if self.scheme == 'asymmetric':
            self.q_x, self.s, self.z = self._quantize_asym(x)
        elif self.scheme == 'symmetric_signed':
            self.q_x, self.s = self._quantize_symm(x, 'signed')
        else:
            self.q_x, self.s = self._quantize_symm(x, 'unsigned')
        return self.q_x, self.s, self.z

    def _quantize_asym(self, x):
        """
        Linear asymmetric quantization method.

        Returns: 
            `q_x`: integer grid
            `s`: scale factor
            `z`: zero point
        """
        qmin = 0.
        qmax = 2.**self.num_bits - 1.
        min_val, max_val = x.min(), x.max()

        s = (max_val - min_val) / (qmax - qmin)

        init_z = qmin - min_val / s

        z = 0
        if init_z < qmin:
            z = qmin
        elif init_z > qmax:
            z = qmax
        else:
            z = init_z

        z = int(z)
        q_x = z + x / s
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        return q_x, s, z

    def _quantize_symm(self, x, sign_type):
        """
        Linear symmetric quantization method.

        Args:
            sign_type: indicate which type to be converted to,
            including `signed` and `unsigned`

        Returns: 
            `q_x`: integer grid
            `s`: scale factor
        """
        if sign_type == 'signed':
            qmin = -2.**(self.num_bits-1)
            qmax = 2.**(self.num_bits-1)-1
        else:
            qmin = 0.
            qmax = 2.**self.num_bits - 1.
        min_val, max_val = x.min(), x.max()

        s = (max_val - min_val) / (qmax - qmin)

        q_x = x / s
        q_x.clamp_(qmin, qmax).round_()
        q_x = q_x.round().byte()
        return q_x, s

    def dequantize(self, q_x, scale, zero_point):
        return scale * (q_x.float() - zero_point)

if __name__ == '__main__':
    print("This module can't run directly!")