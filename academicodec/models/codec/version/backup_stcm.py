import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange 
from compressai.models import CompressionModel
import torch
import torch.nn as nn
import numpy as np
import typing as tp
from typing import Dict, Any
from academicodec.modules import SConv1d
from academicodec.modules import SConvTranspose1d


def conv1d(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, causal: bool = False, norm: str = 'none', norm_kwargs: Dict[str, Any] = {}, pad_mode: str = 'reflect'):
    return SConv1d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel_size,
        stride=stride,
        causal=causal,
        norm=norm,
        norm_kwargs=norm_kwargs,
        pad_mode=pad_mode
    )

def conv_transpose_1d(in_ch: int, out_ch: int, kernel_size: int, stride: int = 1, causal: bool = False, norm: str = 'none', norm_kwargs: Dict[str, Any] = {}, pad_mode: str = 'reflect', trim_right_ratio: float=1.0):
    return SConvTranspose1d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel_size,
        stride=stride,
        causal=causal,
        norm=norm,
        norm_kwargs=norm_kwargs,
        trim_right_ratio=trim_right_ratio
    )


class GDN1D(nn.Module):
    """Example of a Generalized Divisive Normalization (GDN) for 1D data."""
    def __init__(self, num_features, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.eps = 1e-6
        self.beta = nn.Parameter(torch.ones(num_features))
        self.gamma = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        beta = self.beta.view(1, -1, 1)  # (1, num_features, 1)
        gamma = self.gamma.view(1, -1, 1)  # (1, num_features, 1)
        
        if self.inverse:
            # Inverse GDN process
            return x * torch.sqrt(beta + gamma * x**2 + self.eps)
        else:
            # Normal GDN process
            return x / torch.sqrt(beta + gamma * x**2 + self.eps)


class ResidualBlock1D(nn.Module):
    """Simple residual block with two 1D convolutions for audio processing.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv1d(in_ch, out_ch, kernel_size=3)   # padding=1
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv1d(out_ch, out_ch, kernel_size=3)  # , padding=1
        if in_ch != out_ch:
            self.skip = conv1d(in_ch, out_ch, kernel_size=1)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # First convolution
        out = self.conv1(x)
        out = self.leaky_relu(out)

        # Second convolution
        out = self.conv2(out)
        out = self.leaky_relu(out)

        # Skip connection
        if self.skip is not None:
            identity = self.skip(x)

        # Residual connection
        out = out + identity
        return out

# ResidualBlockWithStride1D(N, 2*N, stride=strides[0])
class ResidualBlockWithStride1D(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel_size: int, norm_kwargs: Dict[str, Any] = {}, pad_mode: str = 'reflect'):
        super().__init__()
        self.conv1 = SConv1d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    norm=norm,
                    norm_kwargs=norm_kwargs,
                    causal=causal,
                    pad_mode=pad_mode)
        # self.conv1 = conv1d(in_ch, out_ch, stride=stride, kernel_size=3)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv1d(out_ch, out_ch, kernel_size=3)
        self.gdn = GDN1D(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv1d(in_ch, out_ch, stride=stride, kernel_size=1)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out

class ResidualBlockUpsample1D(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution."""
    def __init__(self, in_ch: int, out_ch: int, stride: int, kernel_size: int):
        super(ResidualBlockUpsample1D, self).__init__()
        self.conv1 = conv_transpose_1d(in_ch, out_ch, stride=stride, kernel_size=kernel_size)  # padding=padding
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv1d(out_ch, out_ch, kernel_size=kernel_size)  # Assuming conv1d_kernel3 is defined elsewhere
        self.gdn = GDN1D(out_ch, inverse=True)  # Assuming GDN1D is defined elsewhere
        if stride != 1 or in_ch != out_ch:
            self.skip = conv_transpose_1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride)    # , padding=padding
        else:
            self.skip = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        # print(f"one{x.shape}")
        out = self.conv1(x)
        # print(f"after conv1{out.shape}")
        out = self.leaky_relu(out)
        # print(f"after leaky_relu{out.shape}")
        out = self.conv2(out)
        # print(f"after conv2{out.shape}")
        out = self.gdn(out)
        # print(f"after gdn{out.shape}")
        # print(out.shape)
        if self.skip is not None:
            identity = self.skip(x)
        # print(f"identity shape:{identity.shape}")
        out += identity
        return out

# 没用到，不过记得之前有报错
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=(kernel_size - 1) // 2,  # 对于奇数kernel_size，这将确保输出长度与输入长度相同
    )

class WMSA(nn.Module):
    # Self-attention module in Swin Transformer
    def __init__(self, input_dim, output_dim, head_dim, window_size, window, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim 
        self.scale = self.head_dim ** -0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.window = 1  # 新增参数 window 用于宽度设置为1
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(torch.zeros((2 * window_size - 1)*(2 * window_size - 1), self.n_heads))

        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=.02)
        self.relative_position_params = torch.nn.Parameter(self.relative_position_params.view(2*window_size-1, 2*window_size-1, self.n_heads).transpose(1, 2).transpose(0, 1))

    def generate_mask(self, h, p, shift):
        """ generating the mask of SW-MSA for 1D sequences
        Args:
            h: number of windows (height)
            p: window size
            shift: shift parameters in CyclicShift.
        Returns:
            attn_mask: mask of shape (1, 1, h, p, p)
        """
        attn_mask = torch.zeros(1, 1, h, p, p, dtype=torch.bool, device=self.relative_position_params.device)
        if self.type == 'W':
            return attn_mask

        s = p - shift
        attn_mask[0, 0, -1, :s, s:] = True  # Current window's left part
        attn_mask[0, 0, -1, s:, :s] = True  # Current window's right part
        attn_mask[0, 0, :-1, :s, s:] = True  # Previous windows' left parts
        attn_mask[0, 0, :-1, s:, :s] = True  # Previous windows' right parts

        return attn_mask



    def forward(self, x):
        """ Forward pass of Window Multi-head Self-attention module.
        Args:
            x: input tensor with shape of [b h w c];
            attn_mask: attention mask, fill -inf where the value is True; 
        Returns:
            output: tensor shape [b h w c]
        """
        # print("进入 forward")
        # # x = x.transpose(1, 2)
        # print(x.shape)
        x = x.unsqueeze(2)
        # print(x.shape)
        # assert False
        
        if self.type != 'W':
            x = torch.roll(x, shifts=(-(self.window_size//2), -(self.window//2)), dims=(1, 2))  # 使用 window 控制宽度的滚动

        x = rearrange(x, 'b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c', p1=self.window_size, p2=self.window)
        h_windows = x.size(1)
        w_windows = x.size(2)
        
        x = rearrange(x, 'b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c', p1=self.window_size, p2=self.window)
        qkv = self.embedding_layer(x)
        q, k, v = rearrange(qkv, 'b nw np (threeh c) -> threeh b nw np c', c=self.head_dim).chunk(3, dim=0)
        sim = torch.einsum('hbwpc,hbwqc->hbwpq', q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), 'h p q -> h 1 1 p q')
        # print(f"sim.shape: {sim.shape}")
        
        if self.type != 'W':
            # attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, self.window, shift=(self.window_size//2, 0))
            attn_mask = self.generate_mask(h_windows, self.window_size, shift=self.window_size//2)
            # attn_mask = self.generate_mask(h_windows, w_windows, self.window_size, shift=self.window//2)
            # print(attn_mask.shape)  # 打印生成的 mask 形状
            # assert False
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum('hbwij,hbwjc->hbwic', probs, v)
        output = rearrange(output, 'h b w p c -> b w p (h c)')
        output = self.linear(output)
        output = rearrange(output, 'b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c', w1=h_windows, p1=self.window_size)

        if self.type != 'W':
            output = torch.roll(output, shifts=(self.window_size//2, self.window//2), dims=(1, 2))

        output = output.squeeze(2)
        return output

    def relative_embedding(self):
        cord = torch.tensor(np.array([[i, j] for i in range(self.window_size) for j in range(self.window)]))
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[:, relation[:, :, 0].long(), relation[:, :, 1].long()]

class Block(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, window, drop_path, type='W', input_resolution=None):
        """ SwinTransformer Block
        """
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        assert type in ['W', 'SW']
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, window, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        # print('first into the block')
        # print(x.shape)
        x = x.transpose(1,2)
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        # print("after the trans_block")
        # print(x.shape)
        x = x.transpose(1,2)
        return x

class ConvTransBlock(nn.Module):
    def __init__(self, conv_dim, trans_dim, head_dim, window_size, window, drop_path, type='W'):
        """ SwinTransformer and Conv Block
        """
        super(ConvTransBlock, self).__init__()
        self.conv_dim = conv_dim
        self.trans_dim = trans_dim
        self.head_dim = head_dim
        self.window_size = window_size
        self.window = window
        self.drop_path = drop_path
        self.type = type
        assert self.type in ['W', 'SW']
        self.trans_block = Block(self.trans_dim, self.trans_dim, self.head_dim, self.window_size, self.window, self.drop_path, self.type)
        self.conv1_1 = nn.Conv1d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, kernel_size=1, stride=1, bias=True)
        self.conv1_2 = nn.Conv1d(self.conv_dim+self.trans_dim, self.conv_dim+self.trans_dim, kernel_size=1, stride=1, bias=True)

        self.conv_block = ResidualBlock1D(self.conv_dim, self.conv_dim)

    def forward(self, x):
        conv_x, trans_x = torch.split(self.conv1_1(x), (self.conv_dim, self.trans_dim), dim=1)
        conv_x = self.conv_block(conv_x) + conv_x
        trans_x = self.trans_block(trans_x)
        res = self.conv1_2(torch.cat((conv_x, trans_x), dim=1))
        x = x + res
        return x

class stcm(CompressionModel):
    def __init__(self, config=[2, 2, 2, 2], strides=[2, 5, 5, 2], head_dim=[8, 16, 16, 8], N=32,  M=512, drop_path_rate=3):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.window_size = 8
        self.window = 1
        dim = N
        self.M = M
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(config))]
        begin = 0
        
        self.m_down1 = nn.Sequential(*[ConvTransBlock(dim//2, dim//2, self.head_dim[0], self.window_size, self.window, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[0])] + \
                      [ResidualBlockWithStride1D(N, 2*N, stride=strides[0])])
        
        self.m_down2 = nn.Sequential(*[ConvTransBlock(dim*2, dim*2, self.head_dim[1], self.window_size, self.window, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[1])] + \
                      [ResidualBlockWithStride1D(4*N, 8*N, stride=strides[1])])

        self.m_up1 = nn.Sequential(*[ConvTransBlock(dim*4, dim*4, self.head_dim[2], self.window_size, self.window, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[2])] + \
                      [ResidualBlockUpsample1D(8*N, 4*N, stride=strides[2],kernel_size=5)])
                      
        self.m_up2 = nn.Sequential(*[ConvTransBlock(dim, dim, self.head_dim[3], self.window_size, self.window, dpr[i+begin], 'W' if not i%2 else 'SW') 
                      for i in range(config[3])] + \
                      [ResidualBlockUpsample1D(2*N, N, stride=strides[3],kernel_size=4)])
                      
        # self.encoderblock = nn.Sequential(*[ResidualBlockWithStride1D(1, 2*N, stride=2)] + self.m_down1 + self.m_down2)
        
        # self.decoderblcok = nn.Sequential(*[ResidualBlockUpsample1D(M, 2*N, upsample=2)] + self.m_up1 + self.m_up2)    

    def encoder2(self, x):
        # y = self.encoderblock(x)
        y1 = self.m_down1(x)
        return y1
        
    def encoder5(self, x):
        y2 = self.m_down2(x)
        return y2

    def decoder5(self, x):
        # y = self.decoderblcok(x)
        y1 = self.m_up1(x)
        return y1
    
    def decoder2(self, x):
        y2 = self.m_up2(x)
        return y2
