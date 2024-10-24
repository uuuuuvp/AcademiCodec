from academicodec.models.encodec.main_net import test_net
from academicodec.models.encodec.net3 import SoundStream
import torch
import librosa
import random
import torchaudio
from einops import rearrange

wav_path = '/data0/youyubo/y/AcademiCodec/sr16/test_16k.wav'
wav, sr = librosa.load(wav_path, sr=16000)
# print(wav[0],wav[1])
# print(wav.shape[0])
# cut speech into proper length
max_len = 16000
if wav.shape[0] > max_len:
    st = random.randint(0, wav.shape[0] - max_len - 1)
    ed = st + max_len
    wav = wav[st:ed]

wav = torch.tensor(wav).unsqueeze(0)
# print(wav.shape)
wav = wav.unsqueeze(1)

""""
# print(wav.shape)

# wav = wav.squeeze(1)
# print(wav.shape)

# wav = wav.squeeze()

# wav = wav.transpose(1,2)
# print(wav.shape)

# wav = wav.transpose(1,2)
# print(wav.shape)

# wav = rearrange(wav, 'b c (l p) -> b l p c', p=5)
# print(wav.shape)
# # wav = rearrange(wav, 'b c (l p) -> b l c p', p=5)
# wav = rearrange(wav, 'b l p c -> b l (p c)')
# num_windows = wav.size(1)
# print(num_windows)
# print(wav.shape)
"""

test_net = test_net(n_filters=32, D=128)
test_net.eval()
test_net(wav)

# sestream = SoundStream(n_filters=32, D=128)
# sestream.eval()
# sestream(wav)



