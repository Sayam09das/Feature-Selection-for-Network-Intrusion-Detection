# =========================================
# PEARSON CORRELATION (LOCAL MULTI-DATASET)
# =========================================

import pandas as pd
import numpy as np
import os

# -----------------------------
# SETTINGS (CHANGE THIS ONLY)
# -----------------------------
DATASET = "TON"   # UNSW / CICIDS / TON

# -----------------------------
# PATH SETUP
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

file_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(file_path)

print(f"\n✅ {DATASET} Dataset Loaded!")
print("Shape:", df.shape)

# -----------------------------
# TARGET DETECTION
# -----------------------------
if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

# -----------------------------
# FEATURES & TARGET
# -----------------------------
Y = df[target_col]

drop_cols = [col for col in ["Label", "Attack"] if col in df.columns]
X = df.drop(columns=drop_cols)

# -----------------------------
# REMOVE NON-NUMERIC (IMPORTANT)
# -----------------------------
X = X.select_dtypes(include=[np.number])

# -----------------------------
# PEARSON CALCULATION
# -----------------------------
pearson_scores = []

for column in X.columns:

    try:
        corr = np.corrcoef(X[column], Y)[0, 1]

        if np.isnan(corr):
            corr = 0

        pearson_scores.append([column, abs(corr)])

    except:
        continue

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
pearson_df = pd.DataFrame(
    pearson_scores,
    columns=["Feature", "Pearson_Correlation"]
)

# -----------------------------
# SORT FEATURES
# -----------------------------
pearson_df = pearson_df.sort_values(
    by="Pearson_Correlation",
    ascending=False
)

# -----------------------------
# SELECT TOP FEATURES
# -----------------------------
k = 20

top_features = pearson_df.head(k)

print("\n🔥 Top Features using Pearson:\n")
print(top_features)

# -----------------------------
# SELECTED FEATURES
# -----------------------------
selected_features = top_features["Feature"].values

print("\n✅ Selected Features:\n")
print(selected_features)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

top_features.to_csv(os.path.join(save_dir, "pearson.csv"), index=False)

with open(os.path.join(save_dir, "pearson_features.txt"), "w") as f:
    f.write(", ".join(selected_features))

print(f"\n📁 Saved at: results/{DATASET}/")