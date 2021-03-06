### Dataset Builder ###

from torch.utils.data import Dataset
from preprocessing import AudioPreProcess

class DatasetAudio(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx]['label']
        audio = AudioPreProcess.load_audio(self.df.iloc[idx, 1])
        rechanneled = AudioPreProcess.audio_to_mono(audio)
        resampled = AudioPreProcess.audio_to_44100(rechanneled)
        resized = AudioPreProcess.audio_to_12sec(resampled)
        shifted = AudioPreProcess.audio_time_shift(resized, shift_range=0.4)
        mel_spec = AudioPreProcess.audio_to_mel(shifted)
        mel_spec_aug = AudioPreProcess.mel_augment(mel_spec, max_mask=0.1, n_freq_masks=1, n_time_masks=1)
        return mel_spec_aug, label

