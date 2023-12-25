# https://github.com/junjun3518/alias-free-torch under the Apache License 2.0

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Aliasing_Free_Activation(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        upsample_ratio: int= 2,
        downsample_ratio: int= 2,
        upsample_kernel_size: int= 12,
        downsample_kernel_size: int= 12,
        snake_alpha: float= 1.0,
        snake_use_log_scale: float= False
        ):
        super().__init__()

        self.upsample = UpSample1d(
            ratio= upsample_ratio,
            kernel_size= upsample_kernel_size,
            )
        self.snake = Snake(
            channels= channels,
            alpha= snake_alpha,
            use_log_scale= snake_use_log_scale
            )
        self.downsample = DownSample1d(
            ratio= downsample_ratio,
            kernel_size= downsample_kernel_size,
            )

    def forward(self, x: torch.FloatTensor):
        x = self.upsample(x)
        x = self.snake(x)
        x = self.downsample(x)

        return x

class Snake(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        alpha: float= 1.0,
        use_log_scale: bool= False
        ):
        super().__init__()
        self.use_log_scale = use_log_scale

        self.alpha = torch.nn.Parameter(torch.empty(channels))
        self.beta = torch.nn.Parameter(torch.empty(channels))

        torch.nn.init.constant_(self.alpha, val= 0.0 if use_log_scale else alpha)
        torch.nn.init.constant_(self.beta, val= 0.0 if use_log_scale else alpha)

    def forward(self, x: torch.FloatTensor):
        alpha = self.alpha[None, :, None]
        beta = self.beta[None, :, None]

        if self.use_log_scale:
            alpha = alpha.exp()
            beta = beta.exp()

        return x + (1.0 / (beta + 1e-5)) * (x * alpha).sin().pow(2.0)



class UpSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        filter = kaiser_sinc_filter1d(cutoff=0.5 / ratio,
                                      half_width=0.6 / ratio,
                                      kernel_size=self.kernel_size)
        self.register_buffer("filter", filter)

    # x: [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.stride, groups=C)
        x = x[..., self.pad_left:-self.pad_right]

        return x

class DownSample1d(nn.Module):
    def __init__(self, ratio=2, kernel_size=None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(cutoff=0.5 / ratio,
                                       half_width=0.6 / ratio,
                                       stride=ratio,
                                       kernel_size=self.kernel_size)

    def forward(self, x):
        xx = self.lowpass(x)

        return xx

def kaiser_sinc_filter1d(cutoff, half_width, kernel_size): # return filter [1,1,kernel_size]
    even = (kernel_size % 2 == 0)
    half_size = kernel_size // 2

    #For kaiser window
    delta_f = 4 * half_width
    A = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if A > 50.:
        beta = 0.1102 * (A - 8.7)
    elif A >= 21.:
        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21.)
    else:
        beta = 0.
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    # ratio = 0.5/cutoff -> 2 * cutoff = 1 / ratio
    if even:
        time = (torch.arange(-half_size, half_size) + 0.5)
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
        # Normalize filter to have sum = 1, otherwise we will have a small leakage
        # of the constant component in the input signal.
        filter_ /= filter_.sum()
        filter = filter_.view(1, 1, kernel_size)

    return filter

class LowPassFilter1d(nn.Module):
    def __init__(self,
                 cutoff=0.5,
                 half_width=0.6,
                 stride: int = 1,
                 padding: bool = True,
                 padding_mode: str = 'replicate',
                 kernel_size: int = 12):
        # kernel_size should be even number for stylegan3 setup,
        # in this implementation, odd number is also possible.
        super().__init__()
        if cutoff < -0.:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = (kernel_size % 2 == 0)
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        filter = kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        self.register_buffer("filter", filter)

    #input [B, C, T]
    def forward(self, x):
        _, C, _ = x.shape

        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right),
                      mode=self.padding_mode)
        out = F.conv1d(x, self.filter.expand(C, -1, -1),
                       stride=self.stride, groups=C)

        return out