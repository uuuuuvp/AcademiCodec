from academicodec.modules.torch_stft import STFT
import torch
import numpy as np
import librosa 
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

audio,_ = librosa.load("/data0/youyubo/y/AcademiCodec/test/dk_16k.wav",sr=16000,duration=4.79)  # 4.79 => 300
device = 'cpu'
filter_length = 1024
hop_length = 256
win_length = 1024 # doesn't need to be specified. if not specified, it's the same as filter_length
window = 'hann'

audio = torch.FloatTensor(audio)
audio = audio.unsqueeze(0)
audio = audio.to(device)

stft = STFT(
    filter_length=filter_length, 
    hop_length=hop_length, 
    win_length=win_length,
    window=window
).to(device)

print(audio.shape)
magnitude, phase = stft.transform(audio)
print(magnitude.shape)
print(phase.shape)
encode=torch.concat((magnitude,phase),dim=0)
print(encode.shape)
# print(encode.unsqueeze(0).shape)
output = stft.inverse(magnitude, phase)
print(output.shape)
print(audio.shape)
# output = output.cpu().data.numpy()[..., :]
# audio = audio.cpu().data.numpy()[..., :]