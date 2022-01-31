#### Data Preprocessing ####

import numpy as np
import torch
import torchaudio
from torchaudio import transforms


### Class for Preprocessing ###
class AudioPreProcess():

    # Load audio file
    @staticmethod
    def load_audio(file_path):
        audio_sig, sample_rate = torchaudio.load(file_path)
        return (audio_sig, sample_rate)

    # Since there are only 28 stereo files and the other files are mono 
    # we can just take the first channel
    @staticmethod
    def audio_to_mono(audio):
        audio_sig, sample_rate = audio
        
        if (audio_sig.shape[0] == 1):
            return audio

        else:
            rec_audio_sig = audio_sig[:1, :]
        
        return ((rec_audio_sig, sample_rate))
    
    # Since we have files with other sample rates than 44100Hz
    # we need to resample the audio signals to 44100Hz
    @staticmethod
    def audio_to_44100(audio):
        audio_sig, sample_rate = audio

        if sample_rate == 44100:
            return audio
        
        resampled_audio = torchaudio.transforms.Resample(sample_rate, 44100)(audio_sig)
        
        return (resampled_audio, 44100)

    # Since 90% of the files have less than 12 seconds of audio
    # we can cut the audio to 12s (when longer than 12s), and pad the ones with less than 12s with zeros
    # @staticmethod
    def audio_to_12sec(audio):
        audio_sig, sample_rate = audio
        audio_sig_length = audio_sig.shape[1]

        # Dividing the sample rate by 1000 and then multiplying it by 12000 - simplify to 12
        max_length = (sample_rate * 120)

        # If the audio is longer than 12s, we cut it to 12s
        if audio_sig_length >= max_length:
            audio_sig = audio_sig[:, :max_length]


        # If the audio is shorter than 12s, we pad it with zeros, half the difference in boths sides
        else:
            padding = max_length - audio_sig_length
            if padding % 2 != 0:
                padding_left = (padding / 2) + 1
                padding_right = (padding / 2)
            else:
                padding_left = padding / 2
                padding_right = padding / 2
            audio_sig = torch.cat((torch.zeros(1, int(padding_left)), audio_sig, torch.zeros(1, int(padding_right))), dim=1)
        
        return (audio_sig, sample_rate)

    # Time shifting the audio to make it more random
    @staticmethod
    def audio_time_shift(audio, shift_range):
        audio_sig, sample_rate = audio
        audio_sig_length = audio_sig.shape[1]

        # Randomly shifting the audio
        shift_amount = int(np.random.uniform(0, 1) * shift_range * audio_sig_length)

        return (audio_sig.roll(shift_amount), sample_rate)

    # Convert to Mel Spectrogram in db
    @staticmethod
    def audio_to_mel(audio, n_mels=64, n_fft=1024, hop_length=None):
        audio_sig, sample_rate = audio

        # Returns Mel Spectrogram with shape [channels, n_mels, time]
        mel_spec = transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)(audio_sig)

        # Convert to db
        db_mel_spec = transforms.AmplitudeToDB(top_db=80)(mel_spec)

        return db_mel_spec

    # Augment Mel Spectrogram on Time and Frequency Axis
    @staticmethod
    def mel_augment(mel_spec, max_mask=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = mel_spec.shape
        mask_value = mel_spec.mean()
        aug_mel_spec = mel_spec.clone()

        # Randomly masking the frequency axis
        freq_mask_param = max_mask * n_mels
        for _ in range(n_freq_masks):
            aug_mel_spec = transforms.FrequencyMasking(freq_mask_param)(aug_mel_spec, mask_value)
        
        # Randomly masking the time axis
        time_mask_param = max_mask * n_steps
        for _ in range(n_time_masks):
            aug_mel_spec = transforms.TimeMasking(time_mask_param)(aug_mel_spec, mask_value)

        return aug_mel_spec



def single_audio_preprocessing(file_path, n_mels=64, n_fft=1024, hop_length=None):
    audio = AudioPreProcess.load_audio(file_path)
    audio = AudioPreProcess.audio_to_mono(audio)
    audio = AudioPreProcess.audio_to_44100(audio)
    audio = AudioPreProcess.audio_to_12sec(audio)
    audio = AudioPreProcess.audio_to_mel(audio, n_mels, n_fft, hop_length)

    return audio

# audio = AudioPreProcess.load_audio('data/backapp_full_audios/502924.wav')
# rechanneled = AudioPreProcess.audio_to_mono(audio)
# resampled = AudioPreProcess.audio_to_44100(rechanneled)
# resized = AudioPreProcess.audio_to_12sec(resampled)
# shifted = AudioPreProcess.audio_time_shift(resized, shift_range=0.4)
# mel_spec = AudioPreProcess.audio_to_mel(shifted)
# mel_spec_aug = AudioPreProcess.mel_augment(mel_spec, max_mask=0.1, n_freq_masks=2, n_time_masks=2)

# print(mel_spec)
# print("Min of waveform: {}\nMax of waveform: {}\nMean of waveform: {}".format(shifted[0].min(), shifted[0].max(), shifted[0].mean()))
# print("Shape of spectrogram: {}".format(mel_spec_aug.size()))

# plt.figure()
# plt.imshow(mel_spec.log2()[0, :, :].numpy())

# print(AudioPreProcess.load_audio('data/backapp_full_audios/492764.wav')[0].shape)

# ------------------------------------------------
# Check for files length
# for index, row in df.iterrows():
#     audio = AudioPreProcess.load_audio(row['filename'])
#     rechanneled = AudioPreProcess.audio_to_mono(audio)
#     resampled = AudioPreProcess.audio_to_44100(rechanneled)
#     resized = AudioPreProcess.audio_to_12sec(resampled)
#     audio_sig, sample_rate = resized

#     print(row['filename'], " - Singal length in seconds", audio_sig.shape[1])

# ------------------------------------------------


# ------------------------------------------------
# # Check for quantiles
# array = []
# for index, row in df.iterrows():
#     audio = AudioPreProcess.load_audio(row['filename'])
#     rechanneled = AudioPreProcess.audio_to_mono(audio)
#     resampled = AudioPreProcess.audio_to_44100(rechanneled)
#     audio_sig, sample_rate = resampled

#     # print(row['filename'], "Singal length in seconds", audio_sig[0].shape[0] / sample_rate)

#     array.append([audio_sig[0].shape[0] / sample_rate])

# print("10th quantile:", np.quantile(array, 0.90))
# ------------------------------------------------


# ------------------------------------------------
# Check for monos and stereos sounds
# Check for sample rates

# mono_count = 0
# stereo_count = 0
# sr_44100 = 0
# sr_other = 0


# for index, row in df.iterrows():
#     audio = AudioPreProcess.load_audio(row['filename'])
#     rechanneled = AudioPreProcess.audio_to_mono(audio)
#     resampled = AudioPreProcess.audio_to_44100(rechanneled)
#     audio_sig, sample_rate = resampled

#     print(row['filename'], audio_sig.shape[0], sample_rate)

#     if audio_sig.shape[0] == 1:
#         mono_count += 1
#         pass
#     else:
#         stereo_count += 1

#     if sample_rate == 44100:
#         sr_44100 += 1
#     else:
#         sr_other += 1
    
# print("Number of mono files:", mono_count)
# print("Number of stereo files:", stereo_count)
# print("Number of 44100Hz Files:", sr_44100)
# print("Number of other Sample Rate Files:", sr_other)


# ------------------------------------------------
