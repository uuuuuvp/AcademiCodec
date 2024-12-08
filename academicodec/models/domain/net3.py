import math
import random
import warnings
import numpy as np
import torch.nn as nn
import torch
from academicodec.modules.torch_stft import STFT
from academicodec.modules.seanet_encoder import SEANetEncoder2d
from academicodec.modules.seanet_decoder import SEANetDecoder2d
from academicodec.quantization import ResidualVectorQuantizer
# from academicodec.modules.tcm import TCM
warnings.filterwarnings("ignore", category=FutureWarning)

# domain model
class SoundStream(nn.Module):
    def __init__(self,
                 n_filters,
                 D,
                 target_bandwidths=[7.5, 15],
                 ratios=[(4, 1), (4, 1), (4, 2), (4, 1)],
                 sample_rate=16000,
                 bins=1024,
                 normalize=False):
        super().__init__()
        self.hop_length = np.prod(ratios)  # 计算乘积
        self.encoder = SEANetEncoder2d(
            n_filters=n_filters, dimension=D, ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] //
                  (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 75
        self.bits_per_codebook = int(math.log2(bins))
        self.target_bandwidths = target_bandwidths
        self.quantizer = ResidualVectorQuantizer(
            dimension=D, n_q=n_q, bins=bins)
        self.decoder = SEANetDecoder2d(
            n_filters=n_filters, ratios=ratios)
        # self.tcm = TCM()
        self.stft = STFT(filter_length=256,hop_length=128, win_length=256,window='hann')
        for param in self.stft.parameters():
            param.requires_grad = False
        
    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def forward(self, x):
        mag, pha = self.stft.transform(x)
        e = torch.concat((mag,pha),dim=1).unsqueeze(1)
        # print(f"进入encode之前{e.shape}")
        e = self.encoder.sedown1(e)
        # print(f"down1:{e.shape}")
        e = self.encoder.sedown2(e)
        # print(f"down2:{e.shape}")
        e = self.encoder.sedown3(e)
        # print(f"down3:{e.shape}")
        e = self.encoder.sedown4(e)
        # print(f"down4:{e.shape}")
        e = self.encoder.sedown5(e)
        # print(f"down5:{e.shape}")        
        max_idx = len(self.target_bandwidths) - 1
        bw = self.target_bandwidths[random.randint(0, max_idx)]
        quantized, codes, bandwidth, commit_loss = self.quantizer(
            e, self.frame_rate, bw)
        # print(quantized.shape)
        o = self.decoder.seup5(quantized)
        # print(f"up5:{o.shape}")
        o = self.decoder.seup4(o)
        # print(f"up4:{o.shape}")
        o = self.decoder.seup3(o)
        # print(f"up3:{o.shape}")
        o = self.decoder.seup2(o)
        # print(f"up2:{o.shape}")
        o = self.decoder.seup1(o)
        # print(f"up1:{o.shape}")
        o = o.squeeze(1)
        mag, pha = torch.split(o, [129, 129], dim=1)
        o = self.stft.inverse(mag, pha)
        o = o.unsqueeze(1)
        return o, commit_loss, None

    def encode(self, x, target_bw=None, st=None):
        mag, pha = self.stft.transform(x)
        e = torch.concat((mag,pha),dim=1).unsqueeze(1)
        # print(f"进入encode之前{e.shape}")
        e = self.encoder.sedown1(e)
        # print(f"down1:{e.shape}")
        e = self.encoder.sedown2(e)
        # print(f"down2:{e.shape}")
        e = self.encoder.sedown3(e)
        # print(f"down3:{e.shape}")
        e = self.encoder.sedown4(e)
        # print(f"down4:{e.shape}")
        e = self.encoder.sedown5(e)
        # print(f"down5:{e.shape}")    
        if target_bw is None:
            bw = self.target_bandwidths[-1]
        else:
            bw = target_bw
        if st is None:
            st = 0
        codes = self.quantizer.encode(e, self.frame_rate, bw, st)
        return codes

    def decode(self, codes):
        quantized = self.quantizer.decode(codes)
        o = self.decoder.seup5(quantized)
        # print(f"up5:{o.shape}")
        o = self.decoder.seup4(o)
        # print(f"up4:{o.shape}")
        o = self.decoder.seup3(o)
        # print(f"up3:{o.shape}")
        o = self.decoder.seup2(o)
        # print(f"up2:{o.shape}")
        o = self.decoder.seup1(o)
        # print(f"up1:{o.shape}")
        o = o.squeeze(1)
        mag, pha = torch.split(o, [129, 129], dim=1)
        o = self.stft.inverse(mag, pha)
        o = o.unsqueeze(1)
        return o