import os
import librosa
import soundfile as sf

def resample_wav_files(directory, output_directory, target_sr):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                input_path = os.path.join(root, file)
                output_path = input_path.replace(directory, output_directory)
                output_dir = os.path.dirname(output_path)
                
                # 创建输出目录（如果不存在）
                os.makedirs(output_dir, exist_ok=True)
                
                # 降采样并保存到新路径
                y, sr = librosa.load(input_path, sr=target_sr, res_type='soxr_vhq')
                sf.write(output_path, y, samplerate=target_sr)
                
                print(f"Resampled {input_path} to {output_path}")

# 设置输入和输出目录
input_directory = r"../../../data/voicebank_demand/data_48k"
output_directory = r"../../../data/voicebank_demand/data_16k"

# 设置目标采样率
target_sample_rate = 16000

os.system(f"rm -rf {output_directory}")
# 调用函数进行降采样
resample_wav_files(input_directory, output_directory, target_sample_rate)
