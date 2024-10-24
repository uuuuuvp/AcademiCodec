# import torchaudio

# input_audio_path = "/data0/youyubo/y/AcademiCodec/sr24/test_24k.wav"
# target_sample_rate = 16000

# # 加载音频文件
# waveform, sample_rate = torchaudio.load(input_audio_path)

# # 重采样
# resampled_waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)

# # 另存为新文件
# output_audio_path = "~/data0/youyubo/y/AcademiCodec/sr16/test.wav"
# torchaudio.save(output_audio_path, resampled_waveform, target_sample_rate)


import soundfile
import librosa
import os

new_sr = 16000
# path = '/mnt/e/data/test_out/input8KHz.wav'
# new_dir_path = '/mnt/e/data/test_out/'

# path = '/data0/youyubo/y/AcademiCodec/sr24/test_24k.wav'
path = '/data0/youyubo/y/AcademiCodec/test/dk.wav'
# new_dir_path = '/mnt/e/data/test_out/'
new_dir_path = './test/'

signal, sr = librosa.load(path, sr = None)

new_signal = librosa.resample(y=signal, orig_sr=sr, target_sr=new_sr)
new_path = os.path.join(new_dir_path, 'dk_16k.wav')

soundfile.write(new_path, new_signal, new_sr)
# print(new_path)