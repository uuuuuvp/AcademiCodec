# 用于测试 net3.py 是否能跑通

from academicodec.models.domain.net3 import SoundStream
import torch
import librosa

wav_path = '/data0/youyubo/y/AcademiCodec/test/dk_16k.wav'
wav, sr = librosa.load(wav_path, sr=16000,duration=4.795/2)

wav = torch.tensor(wav)
wav = wav.unsqueeze(0)
wav = wav.unsqueeze(1)
b,c,l=wav.shape
print(wav.shape)
wav = wav.repeat(10, 1, 1)
print(wav.shape)

test_net = SoundStream(n_filters=32, D=128)
test_net.eval()
test_net(wav)