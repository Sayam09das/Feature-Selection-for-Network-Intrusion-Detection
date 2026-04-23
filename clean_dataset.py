import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("data/NF-UNSW-NB15-v3.csv", low_memory=False)

# -----------------------------
# Separate target
# -----------------------------
target = df["Label"]
df = df.drop(["Label"], axis=1)

# -----------------------------
# Handle missing values
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# Remove duplicates
# -----------------------------
df = df.drop_duplicates()

# -----------------------------
# Drop constant columns
# -----------------------------
df = df.loc[:, df.nunique() > 1]

# -----------------------------
# Encode categorical
# -----------------------------
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# -----------------------------
# Save NON-SCALED dataset (for Dispersion)
# -----------------------------
df["Label"] = target
df.to_csv("NF-UNSW-NB15-v3-NO-SCALE.csv", index=False)

# -----------------------------
# Remove Label again for scaling
# -----------------------------
df = df.drop(["Label"], axis=1)

# -----------------------------
# Handle inf values before scaling
# -----------------------------
# Replace infinity values with NaN, then fill with mean
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# -----------------------------
# Scaling (ONLY features)
# -----------------------------
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# -----------------------------
# Add target back
# -----------------------------
df["Label"] = target

# -----------------------------
# Save SCALED dataset (for Chi-square)
# -----------------------------
df.to_csv("NF-UNSW-NB15-v3-CLEANED.csv", index=False)

print("Both datasets saved successfully!")