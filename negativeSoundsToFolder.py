### Copy 2500 random negative files from the dev folder and 500 from eval folder to the backapp folder ###

import pandas as pd
from shutil import copyfile
import shutil
import os

# main_df = pd.read_csv("data/backapp_full_audios.csv")
# original_dev_df = pd.read_csv("data/FSD50K.ground_truth/dev.csv")
# original_eval_df = pd.read_csv("data/FSD50K.ground_truth/eval.csv")


# dev_path = "data/FSD50K.dev_audio/"
# eval_path = "data/FSD50K.eval_audio/"
# main_path = "data/backapp_full_audios/"

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Create new dataframe with the files to copy and generate csv
def save_new_df(main_df, df_dev, df_eval, main_path):
    new_df = pd.DataFrame(columns=['filepath', 'labels'])
    for i, row in df_dev.iterrows():
        fileId = row['fname']
        filepath = str(fileId) + '.wav'
        file_full_path = main_path + str(fileId) + '.wav'
        if file_full_path not in main_df['filename'].values:
            new_df = new_df.append({'filepath': filepath, 'labels': 0}, ignore_index=True)

    for i, row in df_eval.iterrows():
        fileId = row['fname']
        filepath = str(fileId) + '.wav'
        file_full_path = main_path + str(fileId) + '.wav'
        if file_full_path not in main_df['filename'].values:
            new_df = new_df.append({'filepath': filepath, 'labels': 0}, ignore_index=True)
        
    pd.DataFrame(new_df).to_csv("data/backapp_full_negative_audios.csv", index=False)

    return new_df

# new_df = save_new_df(main_df, original_dev_df, original_eval_df, main_path)
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Copy files from df to the backapp folder
def copy_files(df, orig_path, dest_path):
    new_df = df.sample(n=3000)
    merge_df = pd.DataFrame(columns=['filename', 'label'])
    for i, row in new_df.iterrows():
        filepath = row['filepath']
        copyfile(orig_path + filepath, dest_path + filepath)
        merge_df = merge_df.append({'filename': filepath, 'label': 0}, ignore_index=True)
    pd.DataFrame(merge_df).to_csv("data/backapp_negative_audios.csv", index=False)


# new_df = pd.read_csv("data/backapp_full_negative_audios.csv")

# dest_path = "data/backapp_full_audios/0/"
# orig_path = "data/FSD50K.entire_data/"
# copy_files(new_df, orig_path, dest_path)
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# Copy files from directory to other directory
src_dev = 'data/FSD50K.dev_audio'
src_eval = 'data/FSD50K.eval_audio'
trg = 'data/FSD50K.entire_data'
    

def copy_all_files(src, trg):
# defining source and destination
# paths
    files=os.listdir(src)
    
    # iterating over all the files in
    # the source directory
    for fname in files:
        
        # copying the files to the
        # destination directory
        shutil.copy2(os.path.join(src,fname), trg)

# copy_all_files(src_dev, trg)
# copy_all_files(src_eval, trg)

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# Edit filename from df to only have the filePath
def edit_filename(df):
    for i, row in df.iterrows():
        filepath = row['filename']
        filepath = filepath.split('/')[-1]
        df.at[i, 'filename'] = filepath
    return df

# df = pd.read_csv("data/backapp_full_audios.csv")
# new_df = edit_filename(df)
# new_df.to_csv("data/backapp_positive_audios.csv", index=False)
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#Merge df and create one csv only
def concat_df(df1, df2):
    df = pd.concat([df1, df2])
    df.to_csv("data/backapp_audios.csv", index=False)
    return df

# df1 = pd.read_csv("data/backapp_positive_audios.csv")
# df2 = pd.read_csv("data/backapp_negative_audios.csv")
# df = concat_df(df1, df2)
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# Add file path to df
def add_filepath(df, main_path):
    for i, row in df.iterrows():
        filepath = row['filename']
        label = row['label']
        filepath = main_path + str(label) + '/' + filepath
        df.at[i, 'filepath'] = filepath
        cols = ['filename', 'filepath', 'label']
        df = df[cols]
    return df

main_path = "data/backapp_full_audios/"
df = pd.read_csv("data/backapp_negative_audios.csv")
new_df = add_filepath(df, main_path)
new_df.to_csv("data/backapp_negative_audios_with_path.csv", index=False)