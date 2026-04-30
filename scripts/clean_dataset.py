import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

# -----------------------------
# SETTINGS
# -----------------------------
DATASET_NAME = "TON"

# Resolve paths (works from scripts folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "NF-ToN-IoT-v3.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "cleaned", DATASET_NAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_PATH, low_memory=False)

print("Dataset loaded!")
print("Initial Shape:", df.shape)

# -----------------------------
# AUTO-DETECT TARGET COLUMN
# -----------------------------
if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

print("Target column:", target_col)

# Separate target
target = df[target_col].copy()
df = df.drop(columns=[target_col])

# -----------------------------
# HANDLE MISSING VALUES
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include='object').columns

# numeric
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# categorical
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# REMOVE DUPLICATES
# -----------------------------
df = df.drop_duplicates()

# CRITICAL FIX: target was saved BEFORE duplicate removal
# Reset target to match the cleaned df's index
target = target.loc[df.index].reset_index(drop=True)

# -----------------------------
# DROP CONSTANT COLUMNS
# -----------------------------
df = df.loc[:, df.nunique() > 1]

print("Shape after cleaning:", df.shape)

# -----------------------------
# ENCODE CATEGORICAL
# -----------------------------
le = LabelEncoder()
for col in cat_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# -----------------------------
# SAVE NON-SCALED DATASET
# -----------------------------
df_no_scale = df.copy()
df_no_scale[target_col] = target.values

no_scale_path = os.path.join(OUTPUT_DIR, "no_scale.csv")
df_no_scale.to_csv(no_scale_path, index=False)

print("✅ Saved:", no_scale_path)

# -----------------------------
# SCALING (SAFE VERSION)
# -----------------------------

# Recalculate numeric columns AFTER all processing
num_cols = df.select_dtypes(include=np.number).columns

# Final cleaning before scaling
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# Drop any remaining bad columns (rare case)
valid_cols = df[num_cols].columns[df[num_cols].isnull().sum() == 0]

print("Scaling columns:", len(valid_cols))

# Apply scaling
scaler = StandardScaler()
df[valid_cols] = scaler.fit_transform(df[valid_cols])

# -----------------------------
# ADD TARGET BACK
# -----------------------------
df[target_col] = target.values

# -----------------------------
# FINAL CHECK BEFORE SAVE
# -----------------------------
print("Final shape:", df.shape)
print("Remaining NaN:", df.isnull().sum().sum())

# -----------------------------
# SAVE CLEANED DATASET
# -----------------------------
clean_path = os.path.join(OUTPUT_DIR, "cleaned.csv")

try:
    df.to_csv(clean_path, index=False)
    print("✅ Saved:", clean_path)
except Exception as e:
    print("❌ Error saving cleaned file:", e)

print("\n🚀 Cleaning COMPLETE for CICIDS")