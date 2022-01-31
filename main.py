### Main Running File ###

from model import DistressModel
import torch
import torchaudio
from preprocessing import single_audio_preprocessing

### Load Model
model = DistressModel()
model.load_state_dict(torch.load('output/best_model.pt', map_location='cpu'))
model.eval

### Load Clip and Preprocess
audio_pp = single_audio_preprocessing(file_path='data/FSD50K.dev_audio/443.wav', n_mels=64, n_fft=1024, hop_length=None)
audio_pp = audio_pp.unsqueeze(1)

### Predict
pred = model(audio_pp)
print("Predicted:")
print(f"{pred.argmax(1)}")
