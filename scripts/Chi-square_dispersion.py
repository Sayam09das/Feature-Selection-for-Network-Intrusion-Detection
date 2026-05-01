import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# SETTINGS (CHANGE THIS ONLY)
# -----------------------------
DATASET = "CICIDS"   # UNSW / CICIDS / TON

# -----------------------------
# PATH SETUP
# -----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

scaled_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")
original_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df_scaled = pd.read_csv(scaled_path)
df_original = pd.read_csv(original_path)

print(f"\n✅ {DATASET} Dataset loaded!")

# -----------------------------
# TARGET COLUMN DETECTION
# -----------------------------
if "Label" in df_scaled.columns:
    target_col = "Label"
elif "Attack" in df_scaled.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

# -----------------------------
# FEATURES + TARGET (SAFE DROP)
# -----------------------------
Y = df_scaled[target_col]

# 🔥 SAFE REMOVE (prevents leakage)
drop_cols = [col for col in ["Label", "Attack"] if col in df_scaled.columns]

X_scaled = df_scaled.drop(columns=drop_cols)
X_original = df_original.drop(columns=drop_cols)

print("\nTarget distribution:\n")
print(Y.value_counts())

# -----------------------------
# SCALE FOR CHI-SQUARE (0–1)
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_scaled)
X_scaled = pd.DataFrame(X_scaled, columns=X_original.columns)

# =============================
# CHI-SQUARE
# =============================
k = 20

selector = SelectKBest(score_func=chi2, k=k)
selector.fit(X_scaled, Y)

scores = selector.scores_

feature_scores = pd.DataFrame({
    "Feature": X_original.columns,
    "Score": scores
})

feature_scores = feature_scores.sort_values(by="Score", ascending=False)

print("\n🔥 Top Features using Chi-Square:\n")
print(feature_scores.head(k))

chi_features = feature_scores.head(k)["Feature"].values

print("\n✅ Selected Chi-Square Features:\n")
print(chi_features)

# =============================
# DISPERSION RATIO
# =============================
mean = X_original.mean()
variance = X_original.var()

dispersion_ratio = variance / (mean + 1e-5)

dispersion_df = pd.DataFrame({
    "Feature": X_original.columns,
    "Dispersion": dispersion_ratio
})

dispersion_df = dispersion_df.sort_values(by="Dispersion", ascending=False)

print("\n🔥 Top Features using Dispersion Ratio:\n")
print(dispersion_df.head(k))

dispersion_features = dispersion_df.head(k)["Feature"].values

print("\n✅ Selected Dispersion Features:\n")
print(dispersion_features)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# Save CSV files
feature_scores.head(k).to_csv(os.path.join(save_dir, "chi_square.csv"), index=False)
dispersion_df.head(k).to_csv(os.path.join(save_dir, "dispersion.csv"), index=False)

# Save feature lists
with open(os.path.join(save_dir, "feature_lists.txt"), "w") as f:
    f.write("Chi-Square:\n" + ", ".join(chi_features) + "\n\n")
    f.write("Dispersion:\n" + ", ".join(dispersion_features))

print(f"\n📁 Results saved in: results/{DATASET}/")