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

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

print(f"\n✅ {DATASET} Dataset loaded!")

# -----------------------------
# TARGET COLUMN DETECTION
# -----------------------------
if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

# -----------------------------
# FEATURES + TARGET
# -----------------------------
Y = df[target_col]

drop_cols = [col for col in ["Label", "Attack"] if col in df.columns]
X = df.drop(columns=drop_cols)

print("\nTarget distribution:\n")
print(Y.value_counts())

# =============================
# FISHER SCORE
# =============================
fisher_scores = []
classes = np.unique(Y)

for column in X.columns:

    feature = X[column]
    overall_mean = np.mean(feature)

    numerator = 0
    denominator = 0

    for c in classes:

        class_feature = feature[Y == c]

        class_mean = np.mean(class_feature)
        class_variance = np.var(class_feature)
        n_c = len(class_feature)

        numerator += n_c * ((class_mean - overall_mean) ** 2)
        denominator += n_c * class_variance

    fisher_score = numerator / (denominator + 1e-10)

    fisher_scores.append([column, fisher_score])

# -----------------------------
# CREATE DATAFRAME
# -----------------------------
fisher_df = pd.DataFrame(
    fisher_scores,
    columns=["Feature", "Fisher_Score"]
)

# -----------------------------
# SORT FEATURES
# -----------------------------
fisher_df = fisher_df.sort_values(
    by="Fisher_Score",
    ascending=False
)

# -----------------------------
# SELECT TOP FEATURES
# -----------------------------
k = 20

top_features = fisher_df.head(k)

print("\n🔥 Top Features using Fisher Score:\n")
print(top_features)

fisher_features = top_features["Feature"].values

print("\n✅ Selected Fisher Features:\n")
print(fisher_features)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# Save CSV
top_features.to_csv(
    os.path.join(save_dir, "fisher_score.csv"),
    index=False
)

# Save feature list
with open(os.path.join(save_dir, "feature_lists_fisher.txt"), "w") as f:
    f.write("Fisher:\n" + ", ".join(fisher_features))

print(f"\n📁 Results saved in: results/{DATASET}/")