# ===============================================
# Predictive Maintenance Data Preparation Script
# ===============================================

# ---- Imports ----
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from huggingface_hub import HfApi

# ---- Setup and Constants ----
api_client = HfApi(token=os.getenv("HF_TOKEN"))
OUTPUT_PATH = "Narendran_Predictive/model"
os.makedirs(OUTPUT_PATH, exist_ok=True)

DATA_FILE = "/Narendran_Predictive/data/engine_data.csv"
HF_REPO = "Narendranh/narendran_predictive_data"

# ---- Step 1: Load Dataset ----
try:
    df = pd.read_csv(DATA_FILE)
    df.columns = [
        'Engine_RPM', 'Lub_Oil_Pressure', 'Fuel_Pressure', 'Coolant_Pressure',
        'Lub_Oil_Temperature', 'Coolant_Temperature', 'Engine_Condition'
    ]
    print("[INFO] Dataset successfully loaded and column names standardized.")
except Exception as error:
    print(f"[ERROR] Unable to load dataset from {DATA_FILE}. Details: {error}")

# ---- Step 2: Data Cleaning & Preparation ----
df['Engine_Condition'] = df['Engine_Condition'].astype(int)
print("[INFO] Target variable data type ensured as integer for modeling.")

# ---- Step 3: Feature/Target Split ----
target = "Engine_Condition"
features = df.drop(columns=[target])
labels = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)
print("[INFO] Train-test split completed successfully.")

# ---- Step 4: Save Data Locally ----
train_test_files = {
    "Xtrain.csv": X_train,
    "Xtest.csv": X_test,
    "ytrain.csv": y_train,
    "ytest.csv": y_test
}

for filename, data in train_test_files.items():
    file_path = os.path.join(OUTPUT_PATH, filename)
    data.to_csv(file_path, index=False)
    print(f"[SAVED] {filename} stored at {OUTPUT_PATH}")

# ---- Step 5: Upload to Hugging Face ----
for filename in train_test_files.keys():
    file_path = os.path.join(OUTPUT_PATH, filename)
    try:
        api_client.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=HF_REPO,
            repo_type="dataset"
        )
        print(f"[UPLOAD SUCCESS] {filename} uploaded to Hugging Face Hub.")
    except Exception as error:
        print(f"[UPLOAD FAILED] {filename} could not be uploaded. Error: {error}")
