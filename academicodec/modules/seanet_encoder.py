# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted by Zhihao Du for 2D SEANet

"""Encodec SEANet-based encoder and decoder implementation."""

import typing as tp
import torch
import numpy as np
import torch.nn as nn

from academicodec.modules.funcodec.conv import SConv1d, SConv2d
from academicodec.modules.funcodec.lstm import SLSTM
from academicodec.modules.funcodec.activations import get_activation


# Only channels, norm, causal are different between 24HZ & 48HZ, everything else is default parameter
# 24HZ -> channels = 1, norm = weight_norm, causal = True
# 48HZ -> channels = 2, norm = time_group_norm, causal = False

class SEANetResnetBlock2d(nn.Module):
    """Residual block from SEANet model.
    Args:
        dim (int): Dimension of the input/output
        kernel_sizes (list): List of kernel sizes for the convolutions.
        dilations (list): List of dilations for the convolutions.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3)
        true_skip (bool): Whether to use true skip connection or a simple convolution as the skip connection.
    """
    def __init__(self, dim: int, kernel_sizes: tp.List[tp.Tuple[int, int]] = [(3, 3), (1, 1)],
                 dilations: tp.List[tp.Tuple[int, int]] = [(1, 1), (1, 1)],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, causal: bool = False,
                 pad_mode: str = 'reflect', compress: int = 2, true_skip: bool = True,
                 conv_group_ratio: int = -1):
        super().__init__()
        assert len(kernel_sizes) == len(dilations), 'Number of kernel sizes should match number of dilations'
        # act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)): # this is always length 2
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            # print(in_chs, "_", out_chs) # 32 _ 16; 16 _ 32; 64 _ 32; 32 _ 64; etc until 256 _ 128; 128_ 256 for encode
            block += [
                # act(**activation_params),
                get_activation(activation, **{**activation_params, "channels": in_chs}),
                SConv2d(in_chs, out_chs, kernel_size=kernel_size, dilation=dilation,
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        groups=min(in_chs, out_chs) // 2 // conv_group_ratio if conv_group_ratio > 0 else 1),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut: nn.Module
        # true_skip is always false since the default in SEANetEncoder / SEANetDecoder does not get changed
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv2d(dim, dim, kernel_size=(1, 1), norm=norm, norm_kwargs=norm_params,
                                    causal=causal, pad_mode=pad_mode,
                                    groups=dim // 2 // conv_group_ratio if conv_group_ratio > 0 else 1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x) # This is simply the sum of two tensors of the same size


class ReshapeModule(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        print(f"before reshape{x.shape}")
        return torch.squeeze(x, dim=self.dim)


# Only channels, norm, causal are different between 24HZ & 48HZ, everything else is default parameter
# 24HZ -> channels = 1, norm = weight_norm, causal = True
# 48HZ -> channels = 2, norm = time_group_norm, causal = False
class SEANetEncoder2d(nn.Module):
    """SEANet encoder.
    Args:
        input_size (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order
        activation (str): Activation function. ELU = Exponential Linear Unit
        activation_params (dict): Parameters to provide to the activation function
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
    """
    def __init__(self, input_size: int = 1, dimension: int = 128, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[tp.Tuple[int, int]] = [(4, 1), (4, 1), (4, 2), (4, 1)],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2,
                 seq_model: str = "lstm", seq_layer_num: int = 2, res_seq=True, conv_group_ratio: int = -1):
        super().__init__()
        self.channels = input_size
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        # del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod([x[1] for x in self.ratios])

        # act = getattr(nn, activation)
        mult = 1
        mults = [1, 2, 4, 8, 16]
        down0: tp.List[nn.Module] = [
            SConv2d(input_size, mults[0] * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        self.down0 = nn.Sequential(*down0)
        
        down1: tp.List[nn.Module] = [
            SConv2d(input_size, mults[0] * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)
        ]
        down1 += [SEANetResnetBlock2d(mults[0] * n_filters,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        norm=norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)] 
        down1 += [get_activation(activation, **{**activation_params, "channels": mults[0] * n_filters}),
                  SConv2d(mults[0] * n_filters, mults[0] * n_filters * 2,
                        kernel_size=(ratios[0][0]*2, ratios[0][1]*2),
                        stride=(ratios[0][0], ratios[0][1]),
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        groups=mults[0] * n_filters // 2 // conv_group_ratio if conv_group_ratio > 0 else 1)]
        self.down1 = nn.Sequential(*down1)
        
        down2: tp.List[nn.Module] = [SEANetResnetBlock2d(mults[1] * n_filters,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        norm=norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)]
        # Add downsampling layers
        down2 += [get_activation(activation, **{**activation_params, "channels": mults[1] * n_filters}),
                  SConv2d(mults[1] * n_filters, mults[1] * n_filters * 2,
                        kernel_size=(ratios[1][0]*2, ratios[1][1]*2),
                        stride=(ratios[1][0], ratios[1][1]),
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        groups=mults[1] * n_filters // 2 // conv_group_ratio if conv_group_ratio > 0 else 1)]
        self.down2 = nn.Sequential(*down2)
        
        down3: tp.List[nn.Module] = [SEANetResnetBlock2d(mults[2] * n_filters,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        norm=norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)]
        # Add downsampling layers
        down3 += [get_activation(activation, **{**activation_params, "channels": mults[2] * n_filters}),
                  SConv2d(mults[2] * n_filters, mults[2] * n_filters * 2,
                        kernel_size=(ratios[2][0]*2, ratios[2][1]*2),
                        stride=(ratios[2][0], ratios[2][1]),
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        groups=mults[2] * n_filters // 2 // conv_group_ratio if conv_group_ratio > 0 else 1)]
        self.down3 = nn.Sequential(*down3)
        
        down4: tp.List[nn.Module] = [SEANetResnetBlock2d(mults[3] * n_filters,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        norm=norm, norm_params=norm_params,
                                        activation=activation, activation_params=activation_params,
                                        causal=causal, pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)]
        # Add downsampling layers
        down4 += [get_activation(activation, **{**activation_params, "channels": mults[3] * n_filters}),
                  SConv2d(mults[3] * n_filters, mults[3] * n_filters * 2,
                        kernel_size=(ratios[3][0]*2, ratios[3][1]*2),
                        stride=(ratios[3][0], ratios[3][1]),
                        norm=norm, norm_kwargs=norm_params,
                        causal=causal, pad_mode=pad_mode,
                        groups=mults[3] * n_filters // 2 // conv_group_ratio if conv_group_ratio > 0 else 1)]
        self.down4 = nn.Sequential(*down4)
        

        # squeeze shape for subsequent models
        down5 : tp.List[nn.Module] = [ReshapeModule(dim=2)]

        if seq_model == 'lstm':
            down5 += [SLSTM(mults[4] * n_filters, num_layers=seq_layer_num, skip=res_seq)]
        elif seq_model == "transformer":
            from academicodec.modules.transformer import StreamingTransformerEncoder
            down5 += [StreamingTransformerEncoder(mults[4] * n_filters,
                                         output_size=mults[4] * n_filters,
                                         num_blocks=seq_layer_num,
                                         input_layer=None,
                                         causal_mode="causal" if causal else "None",
                                         skip=res_seq)]
        else:
            pass
        down5 += [get_activation(activation, **{**activation_params, "channels": mults[4] * n_filters}),
            SConv1d(mults[4] * n_filters, dimension,
                    kernel_size=last_kernel_size,
                    norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)]
        self.down5 = nn.Sequential(*down5)

    def sedown0(self, x):
        return self.down0(x)
    
    def sedown1(self, x):
        return self.down1(x)
    
    def sedown2(self, x):
        return self.down2(x)
    
    def sedown3(self, x):
        return self.down3(x)
    
    def sedown4(self, x):
        return self.down4(x)
    
    def sedown5(self, x):
        return self.down5(x)
    
    
    @property
    def input_size(self):
        return self.channels

    def output_size(self):
        return self.dimension

    def forward(self, x):
        # print(f"shouci jinru encode{x.shape}")
        if x.dim() == 3:
            x = x.unsqueeze(1)
        # x in B,C,T, return B,T,C
        # print(f"unsqueeze{x.shape}")
        return self.model(x).permute(0, 2, 1)
