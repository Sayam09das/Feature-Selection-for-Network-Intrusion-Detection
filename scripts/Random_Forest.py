import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# SETTINGS
# -----------------------------
DATASET = "CICIDS"   # UNSW / CICIDS / TON

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

print(f"Dataset loaded: {DATASET}")
print("Original Shape:", df.shape)

# -----------------------------
# CLEAN DATA
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

print("After cleaning:", df.shape)

# -----------------------------
# SAMPLING (IMPORTANT)
# -----------------------------
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

# -----------------------------
# TARGET COLUMN DETECTION
# -----------------------------
if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

Y = df[target_col]

# -----------------------------
# SAFE DROP (IMPORTANT FIX)
# -----------------------------
drop_cols = [
    col for col in ["Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
    if col in df.columns
]
X = df.drop(columns=drop_cols)

# -----------------------------
# TRAIN RANDOM FOREST
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, Y)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
importance = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

# -----------------------------
# TOP FEATURES
# -----------------------------
k = 20

top_features_df = feature_importance_df.head(k)

print("\n🔥 Top Features using Random Forest:\n")
print(top_features_df)

rf_features = top_features_df["Feature"].values

print("\n✅ Selected Random Forest Features:\n")
print(rf_features)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# Save CSV
top_features_df.to_csv(os.path.join(save_dir, "rf_features.csv"), index=False)

# Save TXT
with open(os.path.join(save_dir, "rf_features.txt"), "w") as f:
    f.write(", ".join(rf_features))

print(f"\n📁 Results saved in: results/{DATASET}/")