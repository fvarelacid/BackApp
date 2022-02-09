import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

tokenizer = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


def get_transcription(audio_path):
    #load any audio file of your choice
    speech, rate = librosa.load(audio_path, sr=16000)

    print(speech.shape)


    input_values = tokenizer(speech, return_tensors = 'pt', sampling_rate=16000).input_values

    #Store logits (non-normalized predictions)
    logits = model(input_values).logits


    #Store predicted id's
    predicted_ids = torch.argmax(logits, dim =-1)
    #decode the audio to generate text
    transcriptions = tokenizer.decode(predicted_ids[0])

    return transcriptions


print(get_transcription("data/voice_data/Test1.m4a"))