import torch
import torch.nn as nn
import numpy as np
from numpy.random import RandomState

def complexinit(weights_real, weights_imag, criterion):
    output_chs, input_chs, num_rows, num_cols = weights_real.shape
    fan_in = input_chs
    fan_out = output_chs
    if criterion == 'glorot':
        s = 1. / np.sqrt(fan_in + fan_out) / 4.
    elif criterion == 'he':
        s = 1. / np.sqrt(fan_in) / 4.
    else:
        raise ValueError('Invalid criterion: ' + criterion)

    rng = RandomState()
    kernel_shape = weights_real.shape
    modulus = rng.rayleigh(scale=s, size=kernel_shape)
    phase = rng.uniform(low=-np.pi, high=np.pi, size=kernel_shape)
    weight_real = modulus * np.cos(phase)
    weight_imag = modulus * np.sin(phase)
    weights_real.data = torch.Tensor(weight_real)
    weights_imag.data = torch.Tensor(weight_imag)


# stransconv.py

class DeepSparse(nn.Module):
    def __init__(self, input_chs: int, output_chs: int, num_rows: int, num_cols: int, stride=1, init='he'):
        super(DeepSparse, self).__init__()
        # ... (保持 __init__ 不变) ...
        self.weights_real = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1)))
        self.weights_imag = nn.Parameter(torch.Tensor(1, input_chs, num_rows, int(num_cols // 2 + 1)))
        complexinit(self.weights_real, self.weights_imag, init)
        self.size = (num_rows, num_cols)
        self.stride = stride

    def forward(self, x):
        orig_dtype = x.dtype  # 保存原始 dtype
        x = x.float()  # 强制转 FP32 防止溢出

        # 修正1：增加 norm='ortho' 保持能量守恒，防止数值爆炸
        x = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')

        x_real, x_imag = x.real, x.imag

        # 确保权重也是 float32 (参数可能是 fp16)
        w_real = self.weights_real.float()
        w_imag = self.weights_imag.float()

        # 复数乘法
        y_real = torch.mul(x_real, w_real) - torch.mul(x_imag, w_imag)
        y_imag = torch.mul(x_real, w_imag) + torch.mul(x_imag, w_real)

        # 修正1：对应逆变换也要用 norm='ortho'
        x = torch.fft.irfftn(torch.complex(y_real, y_imag), s=self.size, dim=(-2, -1), norm='ortho')

        # 最后再转回原始精度
        x = x.to(orig_dtype)

        if self.stride == 2:
            x = x[..., ::2, ::2]
        return x

    # ... (loadweight 不变) ...
        
    def loadweight(self, ilayer):
        weight = ilayer.weight.detach().clone()
        fft_shape = self.weights_real.shape[-2]
        weight = torch.flip(weight, [-2, -1])
        pad = torch.nn.ConstantPad2d(padding=(0, fft_shape - weight.shape[-1], 0, fft_shape - weight.shape[-2]),
                             value=0)
        weight = pad(weight)
        weight = torch.roll(weight, (-1, -1), dims=(-2, - 1))
        orig_dtype = filtered.dtype                      # 保存原 dtype
        filtered = filtered.float()                      # 转 float32
        weight_kc = torch.fft.fftn(weight, dim=(-2, -1), norm=None).transpose(0, 1)
        filtered_fft = filtered_fft.type_as(orig_dtype)  # 转回原 dtype
        weight_kc = weight_kc[..., :weight_kc.shape[-1] // 2 + 1]
        self.weights_real.data = weight_kc.real
        self.weights_imag.data = weight_kc.imag
