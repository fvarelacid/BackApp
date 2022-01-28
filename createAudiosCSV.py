#### Access a folder and create a csv file with the audio files in it ####

import os
import pandas as pd

directory_path = "data/backapp_full_audios/"

directory = os.fsencode(directory_path)

df = pd.DataFrame(columns=["filename", "label"])

label = 1

# Iterate through the folder and add file path as a feature (x) and label (y) to the dataframe
for file in os.listdir(directory):
    path_file = os.fsdecode(file)
    df = df.append({"filename": directory_path + path_file, "label": label}, ignore_index=True)

# Convert the dataframe to csv
df.to_csv("data/backapp_full_audios.csv", index=False)