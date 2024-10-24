import librosa

def get_sampling_rate(file_path):
    # 读取音频文件
    y, sr = librosa.load(file_path, sr=None)  # sr=None 以原始采样率加载文件
    return sr

# 替换为你的 FLAC 文件路径
# file_path = "/data0/youyubo/y/SEStream/data/LibriTTS/train-clean-100_16k/4898_28461_000015_000000.wav"
# file_path = "../sr16/test_16k.wav"
file_path = "/data0/youyubo/y/AcademiCodec/test/dk.wav"

sampling_rate = get_sampling_rate(file_path)
print(f"The sampling rate of the FLAC file is: {sampling_rate} Hz")