#### Alg to Copy Sound Files to the Dev Sounds Folder with the right label ####

import pandas as pd
from shutil import copyfile

labels = ["Yell", "Shout", "Screech", "Screaming", "Gunshot_and_gunfire", "Explosion", "Crying_and_sobbing"]

df = pd.read_csv("data/FSD50K.ground_truth/dev.csv")

contain = ''
for label in labels:
    contain += label + '|'

contain = contain[:-1]

df_with_labels = df[df['labels'].str.contains(contain)]

dev_path = "data/FSD50K.dev_audio/"

for index, row in df_with_labels.iterrows():
    file_name = str(row['fname']) + '.wav'
    file_path = dev_path + file_name
    copyfile(file_path, "data/backapp_dev_audio/" + file_name)