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

# codec model
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
        
    def get_last_layer(self):
        return self.decoder.layers[-1].weight
    
    def forward(self, x):
        # x = x.unsqueeze(0)
        # print(x.shape)
        # x = x.unsqueeze(1)
        # print(x.shape)
        mag, pha = self.stft.transform(x)
        # print(mag.shape);print(pha.shape)
        # print('mag and pha shape as above')
        e = torch.concat((mag,pha),dim=1).unsqueeze(1)
        # print(f"进入encode之前{e.shape}")
        # e = self.encoder.forward(e)
        # e = TCM.down1(e)
        # e = self.encoder.sedown0(e)# e = self.en0(x)
        # print(f"down0:{e.shape}")
        # print(e.shape)
        # e = e.permute(0,2,1,3)
        e = self.encoder.sedown1(e)# e = self.en0(x)
        # print(f"down1:{e.shape}")
        e = self.encoder.sedown2(e)# e = self.en0(x)
        # print(f"down2:{e.shape}")
        e = self.encoder.sedown3(e)# e = self.en0(x)
        # print(f"down3:{e.shape}")
        e = self.encoder.sedown4(e)# e = self.en0(x)
        # print(f"down4:{e.shape}")
        e = self.encoder.sedown5(e)# e = self.en0(x)
        # print(f"down5:{e.shape}")
        
        # assert 0
        # print(e.shape)
        max_idx = len(self.target_bandwidths) - 1
        bw = self.target_bandwidths[random.randint(0, max_idx)]
        quantized, codes, bandwidth, commit_loss = self.quantizer(
            e, self.frame_rate, bw)
        # o = quantized
        # print(o.shape)
        # print(quantized.shape)
        o = self.decoder.seup5(quantized)# e = self.en0(x)
        # print(f"up5:{o.shape}")
        o = self.decoder.seup4(o)# e = self.en0(x)
        # print(f"up4:{o.shape}")
        o = self.decoder.seup3(o)# e = self.en0(x)
        # print(f"up3:{o.shape}")
        o = self.decoder.seup2(o)# e = self.en0(x)
        # print(f"up2:{o.shape}")
        o = self.decoder.seup1(o)# e = self.en0(x)
        # print(f"up1:{o.shape}")
        o = o.squeeze(1)
        # mag, pha = o.split()
        mag, pha = torch.split(o, [129, 129], dim=1)
        o = self.stft.inverse(mag, pha)
        o = o.unsqueeze(1)
        # print(o.shape)
        return o, commit_loss, None

    def encode(self, x, target_bw=None, st=None):
        # e = self.encoder(x)
        e = self.encoder.sedown0(x)# e = self.en0(x)
        # print(f"首层大卷积: {e.shape}")
        e = self.stcm.encoder2(e)# e = self.en2(e)
        # print(f"tcm2维卷积: {e.shape}")
        e = self.encoder.sedown2(e)# e = self.en4(e)
        # print(f"seanet卷积: {e.shape}")
        e = self.stcm.encoder5(e)# e = self.en5(e)
        # print(f"tcm5大卷积: {e.shape}")
        e = self.encoder.sedown4(e)# e = self.en8(e)
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
        o = self.decoder.seup5(quantized)# e = self.en0(x)
        # print(f"up5:{o.shape}")
        o = self.decoder.seup4(o)# e = self.en0(x)
        # print(f"up4:{o.shape}")
        o = self.decoder.seup3(o)# e = self.en0(x)
        # print(f"up3:{o.shape}")
        o = self.decoder.seup2(o)# e = self.en0(x)
        # print(f"up2:{o.shape}")
        o = self.decoder.seup1(o)# e = self.en0(x)
        # print(f"up1:{o.shape}")
        o = o.squeeze(1)
        # mag, pha = o.split()
        mag, pha = torch.split(o, [129, 129], dim=1)
        o = self.stft.inverse(mag, pha)
        return o

    def codec(self, x):
        pass

"""
    def forward(self, x):
        # e = self.encoder(x)
        print(x.shape)
        e = self.encoder.sedown0(x)# e = self.en0(x)
        # print(f"首层大卷积: {e.shape}")
        e = self.stcm.encoder2(e)# e = self.en2(e)
        # print(f"tcm2维卷积: {e.shape}")
        e = self.encoder.sedown2(e)# e = self.en4(e)
        # print(f"seanet卷积: {e.shape}")
        e = self.stcm.encoder5(e)# e = self.en5(e)
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
        o = self.stcm.decoder5(o)# o = self.de5(o)
        # print("tcm成功")
        # print(f"tcm5:{o.shape}")
        o = self.decoder.seup2(o)# o = self.de4(o)
        # print(f"seanet4:{o.shape}")
        o = self.stcm.decoder2(o)# o = self.de2(o)
        # print(f"tcm2:{o.shape}")
        o = self.decoder.seup0(o)# o = self.de0(o)
        # print(f"seanet0:{o.shape}")
        # print(x-o)
        return o, commit_loss, None
"""
