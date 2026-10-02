import os
import sys
import zipfile

import pandas as pd # In case of failure !pip install pandas
import gdown # In case of failure !pip install gdown

# Files are public (anyone with link can access)
# the id is the part within /d/ and /view of the public link
ZIP_FILES = {
    "sagi_batch_1.zip" : "1CwAkP8wn6sRA6GSOn_vW8tNzfaB05rTz",
    "sagi_batch_2.zip" : "1y7sI7UM4rxFEr0H8whX0GjlUbXza5YjC",
    "sagi_batch_3.zip" : "1YDGfeFvhQgX3rarcU44Ybctv19VDsmTd"
}

CSV_FILES = {
    "sagi_batch_1.csv" : "1jr-RnE2WrUbyfEtHE2ZXMQrbOC_n3Ind",
    "sagi_batch_2.csv" : "1KHdHMrX20mAdVRgovCCJLygUpY-m1ZZF",
    "sagi_batch_3.csv" : "14fd1JRCa2UzD4ipvsffiQVbniAgdqX7f" 
}


# Destination folder of the dataset
DESTINATION_FOLDER = "dataset/SAGI-D"
os.makedirs(DESTINATION_FOLDER, exist_ok = True)


# Helper function to download files from the drive
def drive_installer(DRIVE_DICTIONARY):
    for file_name, file_id in DRIVE_DICTIONARY.items():
        file_path = os.path.join(DESTINATION_FOLDER, file_name)
        if not os.path.exists(file_path):
            print(f"- Downloading: {file_name}")
            gdown.download(id = file_id, output = file_path, quiet = False)
        else:
            print(f"- {file_name} already present")


# Downloading the dataset ZIP files and csv files
print(f"DATASET DOWNLOAD STARTED\n--> Destination folder: {DESTINATION_FOLDER}")
drive_installer(ZIP_FILES)
drive_installer(CSV_FILES)

# Unpacking the zip files and obtaining the total dataset
for zip_name in ZIP_FILES.keys():
    zip_path = os.path.join(DESTINATION_FOLDER, zip_name)

    # Extracting the zip
    if os.path.exists(zip_path):
        print(f"extracting {zip_name} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Note that all the zip contain the same 2 folders: images_inpaint/ and masks/
            # Extracting the 3 zip files will create the 2 folders merging the content of the 3 batches
            zip_ref.extractall(DESTINATION_FOLDER)
            print(f"- {zip_name} extracted!")
        # Eliminating the original zip file to recover disk space
        print(f"- Eliminating the original zip file to recover disk space")
        os.remove(zip_path)

# Merging the CSV files
print(f"Creating a unique CSV file for the dataset")
df_list = []

for file_name in CSV_FILES.keys():
    csv_path = os.path.join(DESTINATION_FOLDER, file_name)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df_list.append(df)

if df_list:
    final_csv = pd.concat(df_list, ignore_index = True)
    path_final_csv = os.path.join(DESTINATION_FOLDER, "sagi_dataset.csv")
    final_csv.to_csv(path_final_csv, index = False)
    print("    Dataset merged successfully")

    # Eliminating the original csv files to recover disk space
    print("    Eliminating the original csv files to recover disk space")
    for file_name in CSV_FILES.keys():
        csv_path = os.path.join(DESTINATION_FOLDER, file_name)
        if os.path.exists(csv_path):
            os.remove(csv_path)