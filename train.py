### Model Training ###

from torch.utils.data import DataLoader, random_split, ConcatDataset
from dataset import DatasetAudio
import pandas as pd

# Load data df
df = pd.read_csv('data/backapp_full_audios.csv')

# Create 3 different datasets from original df
audio_dataset_1 = DatasetAudio(df, '/Users/franciscovarelacid/Desktop/Strive/BackApp/')
audio_dataset_2 = DatasetAudio(df, '/Users/franciscovarelacid/Desktop/Strive/BackApp/')
audio_dataset_3 = DatasetAudio(df, '/Users/franciscovarelacid/Desktop/Strive/BackApp/')

audio_dataset = ConcatDataset([audio_dataset_1, audio_dataset_2, audio_dataset_3])

# Define the number of samples for training and validation - 80% and 20% respectively
train_size = int(0.8 * len(audio_dataset))
val_size = len(audio_dataset) - train_size

# Randomly split the dataset into training and validation sets
train_dataset, val_dataset = random_split(audio_dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)