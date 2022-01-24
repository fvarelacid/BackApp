#### Data Preprocessing ####
import pandas as pd

labels = ["Yell", "Shout", "Screech", "Screaming", "Gunshot_and_gunfire", "Explosion", "Crying_and_sobbing"]

df = pd.read_csv("data/FSD50K.ground_truth/dev.csv")

print(df.head())