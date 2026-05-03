import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# -----------------------------
# SETTINGS (CHANGE THIS ONLY)
# -----------------------------
DATASET = "CICIDS"   # UNSW / CICIDS / TON
k = 20               # number of features to select
sample_size = 30000  # IMPORTANT for speed

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
# CLEANING
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)

# -----------------------------
# SAMPLING (VERY IMPORTANT)
# -----------------------------
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

print(f"\nAfter sampling: {df.shape}")

# -----------------------------
# TARGET COLUMN
# -----------------------------
if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

Y = df[target_col]

drop_cols = [col for col in ["Label", "Attack"] if col in df.columns]
X = df.drop(columns=drop_cols)

# -----------------------------
# REMOVE CONSTANT FEATURES
# -----------------------------
X = X.loc[:, X.nunique() > 1]

print(f"After cleaning: {X.shape}")

# =============================
# FORWARD SELECTION
# =============================
model = LogisticRegression(max_iter=500, solver='liblinear')

selected_features = []
remaining_features = list(X.columns)

scores_list = []

for i in range(k):

    best_score = -1
    best_feature = None

    print(f"\n🔄 Selecting feature {i+1}/{k}...")

    for feature in remaining_features:

        temp_features = selected_features + [feature]

        try:
            scores = cross_val_score(
                model,
                X[temp_features],
                Y,
                cv=2,
                scoring="accuracy",
                n_jobs=-1
            )

            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_feature = feature

        except:
            continue

    if best_feature is None:
        break

    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    scores_list.append(best_score)

    print(f"✅ Selected: {best_feature} | Score: {best_score:.5f}")

# -----------------------------
# FINAL OUTPUT
# -----------------------------
forward_df = pd.DataFrame({
    "Feature": selected_features,
    "Score": scores_list
})

print("\n🔥 Final Forward Selected Features:\n")
print(forward_df)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

forward_df.to_csv(os.path.join(save_dir, "forward.csv"), index=False)

with open(os.path.join(save_dir, "feature_lists_forward.txt"), "w") as f:
    f.write(", ".join(selected_features))

print(f"\n📁 Forward results saved in: results/{DATASET}/")