### Algorithm to Convert Sounds to Wav ###

import os

directory = os.fsencode("data/sounds_to_wav/")

### Convert Non Wav Files to Wav ###
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mp3") or filename.endswith(".aiff") or filename.endswith(".flac"):
        print("Converting", filename, "to wav...")
        os.system("ffmpeg -i data/sounds_to_wav/" + filename + " -ac 1 -ar 16000 data/sounds_to_wav/" + filename[:-4] + ".wav")

### Delete Non Wav Files ###
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".wav") == False:
        print("Deleting", filename)
        os.remove("data/sounds_to_wav/" + filename)
        print("File", filename, "deleted.")

