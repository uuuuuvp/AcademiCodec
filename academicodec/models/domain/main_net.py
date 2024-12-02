import math
import random

import numpy as np
import torch.nn as nn
from academicodec.modules.seanet import SEANetDecoder1
from academicodec.modules.seanet import SEANetEncoder1
from academicodec.quantization import ResidualVectorQuantizer
from academicodec.modules.stcm import stcm

# Generator
class tcnet(nn.Module):
    def __init__(self,
                 n_filters,
                 D,
                 target_bandwidths=[7.5, 15],
                 ratios=[8, 5, 4, 2],
                 sample_rate=16000,
                 bins=1024,
                 normalize=False):
        super().__init__()
        self.hop_length = np.prod(ratios)  # 计算乘积
        self.encoder = SEANetEncoder1(
            n_filters=n_filters, dimension=D, ratios=ratios)
        n_q = int(1000 * target_bandwidths[-1] //
                  (math.ceil(sample_rate / self.hop_length) * 10))
        self.frame_rate = math.ceil(sample_rate / np.prod(ratios))  # 75
        self.bits_per_codebook = int(math.log2(bins))
        self.target_bandwidths = target_bandwidths
        self.quantizer = ResidualVectorQuantizer(
            dimension=D, n_q=n_q, bins=bins)
        self.decoder = SEANetDecoder1(
            n_filters=n_filters, dimension=D, ratios=ratios)
        self.tcm = stcm()



    def get_last_layer(self):
        return self.decoder.layers[-1].weight

    def forward(self, x):
        # e = self.encoder(x)
        e = self.encoder.sedown0(x)# e = self.en0(x)
        # print(f"首层大卷积: {e.shape}")
        e = self.tcm.encoder2(e)# e = self.en2(e)
        # print(f"tcm2维卷积: {e.shape}")
        e = self.encoder.sedown2(e)# e = self.en4(e)
        # print(f"seanet卷积: {e.shape}")
        e = self.tcm.encoder5(e)# e = self.en5(e)
        # print(f"tcm5大卷积: {e.shape}")
        e = self.encoder.sedown4(e)# e = self.en8(e)
        # print(f"seanet卷积: {e.shape}") 
        max_idx = len(self.target_bandwidths) - 1
        bw = self.target_bandwidths[random.randint(0, max_idx)]
        quantized, codes, bandwidth, commit_loss = self.quantizer(
            e, self.frame_rate, bw)
        # print('\n')
        # print(f"量化结果{quantized.shape}")
        # o = self.decoder(quantized)
        o = self.decoder.seup4(quantized)# o = self.de8(quantized)
        # print(f"反卷积8步{o.shape}")
        o = self.tcm.decoder5(o)# o = self.de5(o)
        # print("tcm成功")
        # print(f"tcm5:{o.shape}")
        o = self.decoder.seup2(o)# o = self.de4(o)
        # print(f"seanet4:{o.shape}")
        o = self.tcm.decoder2(o)# o = self.de2(o)
        # print(f"tcm2:{o.shape}")
        o = self.decoder.seup0(o)# o = self.de0(o)
        # print(f"seanet0:{o.shape}")
        # print(x-o)
        return o, commit_loss, None

    # main_net 只是用了forward，但是没有分别写出 encode decode 
    # 主要还是借助原本的框架 net3 , 尤其是 test 在 net3 里实现
    def encode(self, x, target_bw=None, st=None):
        e = self.encoder(x)
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
        o = self.decoder(quantized)
        return o

    def codec(self, x):
        pass
