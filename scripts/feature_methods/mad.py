import numpy as np
import pandas as pd
import os

# -----------------------------
# SETTINGS (CHANGE THIS ONLY)
# -----------------------------
DATASET = "UNSW"   # UNSW / CICIDS / TON

# -----------------------------
# PATH SETUP
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

print(f"\n{DATASET} Dataset loaded!")

# -----------------------------
# TARGET COLUMN DETECTION
# -----------------------------
if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("No target column found!")

# -----------------------------
# FEATURES + TARGET
# -----------------------------
Y = df[target_col]

drop_cols = [col for col in ["Label", "Attack"] if col in df.columns]
X = df.drop(columns=drop_cols)

# -----------------------------
# CLEANING
# -----------------------------
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

# -----------------------------
# REMOVE CONSTANT FEATURES
# -----------------------------
X = X.loc[:, X.nunique() > 1]

print(f"\nAfter cleaning: {X.shape}")

# =============================
# MAD (Mean Absolute Deviation)
# =============================
mad_scores = {}

for col in X.columns:
    mad_scores[col] = np.mean(np.abs(X[col] - np.mean(X[col])))

mad_df = pd.DataFrame({
    "Feature": mad_scores.keys(),
    "Score": mad_scores.values()
})

mad_df = mad_df.sort_values(by="Score", ascending=False)

# -----------------------------
# SELECT TOP K
# -----------------------------
k = 20
mad_top = mad_df.head(k)

print("\nTop Features using MAD:\n")
print(mad_top)

mad_features = mad_top["Feature"].values

print("\nSelected MAD Features:\n")
print(mad_features)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# Save CSV
mad_top.to_csv(os.path.join(save_dir, "mad.csv"), index=False)

# Save feature list
with open(os.path.join(save_dir, "feature_lists_mad.txt"), "w") as f:
    f.write(", ".join(mad_features))

print(f"\nMAD results saved in: results/{DATASET}/")
