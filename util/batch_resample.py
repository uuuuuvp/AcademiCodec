import os
import librosa
import soundfile
# 奇怪的权限
# 定义转换采样率的函数，接收3个变量：原音频路径、重新采样后的音频存储路径、目标采样率
def change_sample_rate(path, new_dir_path, new_sample_rate):
    wavfile = path.split('/')[-1]  # 提取音频文件名，如“1.wav"
    # new_file_name = wavfile.split('.')[0] + '_8k.wav'      #此行代码可用于对转换后的文件进行重命名（如有需要）

    signal, sr = librosa.load(path,sr=None)  # 调用librosa载入音频

    new_signal = librosa.resample(y=signal, orig_sr=sr, target_sr=new_sample_rate)  # 调用librosa进行音频采样率转换
    

    new_path = os.path.join(new_dir_path,wavfile)  # 指定输出音频的路径，音频文件与原音频同名
    # new_path = os.path.join(new_dir_path, new_file_name)      #若需要改名则启用此行代码
    print(new_path)
    # sr_2=add(sr)
    #new_signal_2=add(new_signal)

    #librosa.output.write_wav(new_path, new_signal , new_sample_rate)      #因版本问题，此方法可能用不了
    soundfile.write(new_path,new_signal, new_sample_rate)
    return
    #soundfile.write(new_path,new_signal, new_sample_rate)


if __name__ == '__main__':
    # 指定原音频文件夹路径
    # original_path = "/mnt/e/code/data/dataset/welsh_english_male"
    original_path = "/data0/youyubo/y/data/test-clean_8k"
    wav_list = os.listdir(original_path)

    # 指定转换后的音频文件夹路径
    # new_dir_path = "/mnt/e/code/data/resample/wel-m"
    new_dir_path = "/data0/youyubo/y/data/test-clean-16k"
    os.makedirs(new_dir_path, exist_ok=True)

    # 开始以对原音频文件夹内的音频进行采样率的批量转换
    for i in wav_list:
        wav_path = os.path.join(original_path, i)
        change_sample_rate(wav_path, new_dir_path, 16000)

