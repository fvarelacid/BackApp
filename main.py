### Main Running File ###

from model import DistressModel
import torch
import torchaudio
from preprocessing import single_audio_preprocessing
# from huggingFace import get_transcription

### Load Model
model = DistressModel()
model.load_state_dict(torch.load('output/best_model_2.pt', map_location='cpu'))
model.eval

### Load Clip and Preprocess
audio_pp = single_audio_preprocessing(file_path='data/FSD50K.entire_data/61912.wav', n_mels=64, n_fft=1024, hop_length=None)
X = audio_pp
X_m, X_s = X.mean(), X.std()
X = (X - X_m) / X_s
X = X.unsqueeze(1)


### Predict
pred = model(X)
print("Predicted:")
print(f"{pred.argmax(1)}")


# ------------------------------------------------------- #
### Transcript Prediction ###

# ### Words for help
# words = [["LEAVE", "ME", "ALONE"], ["DON'T", "TOUCH", "ME"], ["GET", "AWAY", "FROM", "ME"], ["PLEASE", "DON'T", "DO", "IT"], 
# ["GET", "OFF"], ["STOP", "IT"], ["STOP"], ["HELP", "ME"], ["HELP"], ["DON'T"],
# ["WHAT", "YOU" ,"DOING"], ["GO", "AWAY"], ["LET", "ME", "GO"], ["PLEASE", "STOP"], ["PLEASE", "STOP", "IT"]]


# print(get_transcription(audio_path))
