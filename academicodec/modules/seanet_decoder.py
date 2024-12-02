import typing as tp
import numpy as np
import torch.nn as nn
import torch
from academicodec.modules.funcodec.conv import SConv1d, SConv2d
from academicodec.modules.funcodec.conv import SConvTranspose1d, SConvTranspose2d
from academicodec.modules.funcodec.lstm import SLSTM
from academicodec.modules.funcodec.activations import get_activation


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
        return torch.unsqueeze(x, dim=self.dim)


class SEANetDecoder2d(nn.Module):
    """SEANet decoder.
    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function
        final_activation (str): Final activation function after all convolutions.
        final_activation_params (dict): Parameters to provide to the activation function
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
        trim_right_ratio (float): Ratio for trimming at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
    """
    def __init__(self, input_size: int = 128, channels: int = 1, n_filters: int = 32, n_residual_layers: int = 1,
                 ratios: tp.List[tp.Tuple[int, int]] = [(4, 1), (4, 1), (4, 2), (4, 1)],
                 activation: str = 'ELU', activation_params: dict = {'alpha': 1.0},
                 final_activation: tp.Optional[str] = None, final_activation_params: tp.Optional[dict] = None,
                 norm: str = 'weight_norm', norm_params: tp.Dict[str, tp.Any] = {}, kernel_size: int = 7,
                 last_kernel_size: int = 7, residual_kernel_size: int = 3, dilation_base: int = 2, causal: bool = False,
                 pad_mode: str = 'reflect', true_skip: bool = False, compress: int = 2,
                 seq_model: str = 'lstm', seq_layer_num: int = 2, trim_right_ratio: float = 1.0, res_seq=True,
                 last_out_padding: tp.List[tp.Union[int, int]] = [(0, 1), (0, 0)],
                 tr_conv_group_ratio: int = -1, conv_group_ratio: int = -1):
        super().__init__()
        self.dimension = input_size
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        # del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod([x[1] for x in self.ratios])

        # act = getattr(nn, activation)
        # mult = int(2 ** len(self.ratios))
        mults = [1, 2, 4, 8, 16]
        
        up5: tp.List[nn.Module] = [SConv1d(input_size, mults[4] * n_filters, kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)]
        if seq_model == "lstm":
            up5 += [SLSTM(mults[4] * n_filters, num_layers=seq_layer_num, skip=res_seq)]
        elif seq_model == "transformer":
            from academicodec.modules.transformer import StreamingTransformerEncoder
            up5 += [StreamingTransformerEncoder(mults[4] * n_filters,
                                         output_size=mults[4] * n_filters,
                                         num_blocks=seq_layer_num,
                                         input_layer=None,
                                         causal_mode="causal" if causal else "None",
                                         skip=res_seq)]
        else:
            pass
        up5 += [ReshapeModule(dim=2)]
        self.up5 = nn.Sequential(*up5)

        # Add upsampling layers
        up4 : tp.List[nn.Module] =  [get_activation(activation, **{**activation_params, "channels": mults[4] * n_filters}),
            SConvTranspose2d(mults[4] * n_filters, mults[4] * n_filters // 2,
                                 kernel_size=(ratios[3][0] * 2, ratios[3][1] * 2),
                                 stride=(ratios[3][0], ratios[3][1]),
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio,
                                 out_padding=last_out_padding if 0 == len(self.ratios) - 1 else 0,
                                 groups=mults[4] * n_filters // 2 // tr_conv_group_ratio if tr_conv_group_ratio > 0 else 1)
        ]
        # Add residual layers

        up4 += [get_activation(activation, **{**activation_params, "channels": mults[3] * n_filters}),
            SEANetResnetBlock2d(mults[4] * n_filters // 2,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        activation=activation, activation_params=activation_params,
                                        norm=norm, norm_params=norm_params, causal=causal,
                                        pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)
            ]
        self.up4 = nn.Sequential(*up4)

        up3 : tp.List[nn.Module] =  [get_activation(activation, **{**activation_params, "channels": mults[3] * n_filters}),
            SConvTranspose2d(mults[3] * n_filters, mults[3] * n_filters // 2,
                                 kernel_size=(ratios[2][0] * 2, ratios[2][1] * 2),
                                 stride=(ratios[2][0], ratios[2][1]),
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio,
                                 out_padding=last_out_padding if 1 == len(self.ratios) - 1 else 0,
                                 groups=mults[3] * n_filters // 2 // tr_conv_group_ratio if tr_conv_group_ratio > 0 else 1)
        ]
        # Add residual layers

        up3 += [get_activation(activation, **{**activation_params, "channels": mults[2] * n_filters}),
            SEANetResnetBlock2d(mults[3] * n_filters // 2,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        activation=activation, activation_params=activation_params,
                                        norm=norm, norm_params=norm_params, causal=causal,
                                        pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)
            ]
        self.up3 = nn.Sequential(*up3)

        up2 : tp.List[nn.Module] =  [get_activation(activation, **{**activation_params, "channels": mults[2] * n_filters}),
            SConvTranspose2d(mults[2] * n_filters, mults[2] * n_filters // 2,
                                 kernel_size=(ratios[1][0] * 2, ratios[1][1] * 2),
                                 stride=(ratios[1][0], ratios[1][1]),
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio,
                                 out_padding=last_out_padding if 2 == len(self.ratios) - 1 else 0,
                                 groups=mults[2] * n_filters // 2 // tr_conv_group_ratio if tr_conv_group_ratio > 0 else 1)
        ]
        # Add residual layers

        up2 += [get_activation(activation, **{**activation_params, "channels": mults[1] * n_filters}),
            SEANetResnetBlock2d(mults[2] * n_filters // 2,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        activation=activation, activation_params=activation_params,
                                        norm=norm, norm_params=norm_params, causal=causal,
                                        pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)
            ]
        self.up2 = nn.Sequential(*up2)

        up1 : tp.List[nn.Module] = [get_activation(activation, **{**activation_params, "channels": mults[0] * n_filters}),
            SConvTranspose2d(mults[1] * n_filters, mults[1] * n_filters // 2,
                                 kernel_size=(ratios[0][0] * 2, ratios[0][1] * 2),
                                 stride=(ratios[0][0], ratios[0][1]),
                                 norm=norm, norm_kwargs=norm_params,
                                 causal=causal, trim_right_ratio=trim_right_ratio,
                                 out_padding=last_out_padding if 3 == len(self.ratios) - 1 else 0,
                                 groups=mults[1] * n_filters // 2 // tr_conv_group_ratio if tr_conv_group_ratio > 0 else 1)
        ]
        # Add residual layers

        up1 += [get_activation(activation, **{**activation_params, "channels": mults[0] * n_filters}),
            SEANetResnetBlock2d(mults[1] * n_filters // 2,
                                        kernel_sizes=[(residual_kernel_size, residual_kernel_size), (1, 1)],
                                        dilations=[(1, dilation_base ** 1), (1, 1)],
                                        activation=activation, activation_params=activation_params,
                                        norm=norm, norm_params=norm_params, causal=causal,
                                        pad_mode=pad_mode, compress=compress, true_skip=true_skip,
                                        conv_group_ratio=conv_group_ratio)
            ]
        up1 += [get_activation(activation, **{**activation_params, "channels": n_filters}),
            SConv2d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)]
        self.up1 = nn.Sequential(*up1)
        
        up0 : tp.List[nn.Module] = [get_activation(activation, **{**activation_params, "channels": n_filters}),
            SConv2d(n_filters, channels, last_kernel_size, norm=norm, norm_kwargs=norm_params,
                    causal=causal, pad_mode=pad_mode)]
        self.up0 = nn.Sequential(*up0)
        
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None: # This is always None
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model : tp.List[nn.Module] = [
                final_act(**final_activation_params)
            ]
            self.model = nn.Sequential(*model)

    def output_size(self):
        return self.channels

    def forward(self, z):
        # z in (B, T, C)
        y = self.model(z.permute(0, 2, 1))
        return y

    def seup5(self, x):
        return self.up5(x)
    
    def seup4(self, x):
        return self.up4(x)
    
    def seup3(self, x):
        return self.up3(x)
    
    def seup2(self, x):
        return self.up2(x)
    
    def seup1(self, x):
        return self.up1(x)
    
    def seup0(self, x):
        return self.up0(x)