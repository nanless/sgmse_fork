### use scipy to downsample all .wav audio files in a folder to 16kHz sampling rate and save them in a new folder

import os
import scipy.io.wavfile as wav
import numpy as np
import librosa
import soundfile as sf

# path to the folder containing the audio files
path = '/data2/zhounan/data/noisy/voicebank_demand/sgmse_data/valid/noisy/'

# path to the folder where the downsampled audio files will be saved
save_path = '/data2/zhounan/data/noisy/voicebank_demand/sgmse_data/valid_16k/noisy/'

# create the folder if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

# loop through all files in the folder
for filename in os.listdir(path):
    if filename.endswith('.wav'):
        # load the audio file
        audio, sr = librosa.load(path+filename, sr=None)
        # downsample the audio file to 16kHz
        audio = librosa.resample(audio, sr, 16000, res_type='fft')
        # save the downsampled audio file
        sf.write(save_path+filename, audio, 16000)