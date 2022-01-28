#### Data Preprocessing ####

import numpy as np
import pandas as pd
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio


### Load Data ###
df = pd.read_csv('data/backapp_full_audios.csv')

### Class for Preprocessing ###
class AudioPreProcess():

    # Load audio file
    def load_audio(file_path):
        audio_sig, sample_rate = torchaudio.load(file_path)
        return audio_sig, sample_rate

    # Since there are only 28 stereo files and the other files are mono 
    # we can just take the first channel
    def audio_to_mono(audio):
        audio_sig, sample_rate = audio
        if audio_sig.shape[0] == 2:
            audio_sig = audio_sig[:1, :]
        return audio_sig, sample_rate
    
    # Since we have files with other sample rates than 44100Hz
    # we need to resample the audio signals to 44100Hz
    def audio_to_44100(audio):
        audio_sig, sample_rate = audio
        if sample_rate != 44100:
            sample_rate = torchaudio.transforms.Resample(sample_rate, 44100)(audio_sig)
        return audio_sig, sample_rate



# ------------------------------------------------
# Check for monos and stereos sounds
# Check for sample rates

mono_count = 0
stereo_count = 0
sr_44100 = 0
sr_other = 0


for index, row in df.iterrows():
    audio = AudioPreProcess.load_audio(row['filename'])
    audio = AudioPreProcess.audio_to_mono(audio)
    audio_sig, sample_rate = AudioPreProcess.audio_to_44100(audio)

    print(row['filename'], audio_sig, type(sample_rate))

    if audio_sig[0] == 1:
        mono_count += 1
        pass
    else:
        stereo_count += 1

    if sample_rate == 44100:
        sr_44100 += 1
    else:
        sr_other += 1
    
print("Number of mono files:", mono_count)
print("Number of stereo files:", stereo_count)
print("Number of 44100Hz Files:", sr_44100)
print("Number of other Sample Rate Files:", sr_other)


# ------------------------------------------------
