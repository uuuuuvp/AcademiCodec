import os

def create_lst_file(root_folder, output_file):
    with open(output_file, 'w') as lst_file:
        for foldername, subfolders, filenames in os.walk(root_folder):
            # 遍历文件夹中的所有文件
            for filename in filenames:
                if filename.endswith('.flac'):
                    # 获取文件的完整路径（绝对路径）
                    file_path = os.path.join(foldername, filename)
                    
                    # 将绝对路径写入 .lst 文件
                    lst_file.write(f"{file_path}\n")

if __name__ == "__main__":
    root_folder = "/data0/youyubo/wang/LbriSpeech/LibriSpeech/train-clean-100/"  # 根文件夹的路径
    output_file = "./train-clean-100.lst"  # 输出的 .lst 文件名
    create_lst_file(root_folder, output_file)
