import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

DATASET_NAME = "TON"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "NF-ToN-IoT-v3.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "cleaned", DATASET_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Dataset loaded!")
print("Initial Shape:", df.shape)

if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("No target column found!")

print("Target column:", target_col)

target = df[target_col].copy()
df = df.drop(columns=[target_col])

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

target = target.loc[df.index].reset_index(drop=True)

df = df.loc[:, df.nunique() > 1]

print("Shape after cleaning:", df.shape)

le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

df_no_scale = df.copy()
df_no_scale[target_col] = target.values

no_scale_path = os.path.join(OUTPUT_DIR, "no_scale.csv")
df_no_scale.to_csv(no_scale_path, index=False)

print("Saved:", no_scale_path)

num_cols = df.select_dtypes(include=np.number).columns

df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

valid_cols = df[num_cols].columns[df[num_cols].isnull().sum() == 0]

print("Scaling columns:", len(valid_cols))

scaler = StandardScaler()
df[valid_cols] = scaler.fit_transform(df[valid_cols])

df[target_col] = target.values

print("Final shape:", df.shape)
print("Remaining NaN:", df.isnull().sum().sum())

clean_path = os.path.join(OUTPUT_DIR, "cleaned.csv")

try:
    df.to_csv(clean_path, index=False)
    print("Saved:", clean_path)
except Exception as e:
    print("Error saving cleaned file:", e)

print("\nCleaning COMPLETE for CICIDS")
