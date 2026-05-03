import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# -----------------------------
# SETTINGS
# -----------------------------
DATASET = "UNSW"   # UNSW / CICIDS / TON

# -----------------------------
# PATH
# -----------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

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
# REMOVE LEAKAGE COLUMNS
# -----------------------------
leak_cols = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "DNS_QUERY_ID",
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS"
]

df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

# -----------------------------
# SAMPLE
# -----------------------------
sample_size = min(30000, len(df))
df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

# -----------------------------
# CLEAN DATA
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna().reset_index(drop=True)

print("After cleaning:", df.shape)

# -----------------------------
# FEATURES + TARGET
# -----------------------------
Y = df[target_col]

# IMPORTANT FIX:
# remove BOTH target columns so Attack/Label never becomes a feature
X = df.drop(columns=[target_col, "Attack", "Label"], errors="ignore")

# keep only numeric columns
X = X.select_dtypes(include=[np.number])

# remove constant features
X = X.loc[:, X.nunique() > 1]

print("Feature shape:", X.shape)
print("Target distribution:\n", Y.value_counts())

# -----------------------------
# FORWARD SELECTION
# -----------------------------
model = LogisticRegression(
    max_iter=5000,
    solver="liblinear"
)

selected = []
remaining = list(X.columns)

k = min(20, len(remaining))
scores_log = []

for i in range(k):

    print(f"\nSelecting feature {i + 1}/{k}...")

    best_score = -1
    best_feature = None

    for feature in remaining:
        temp_features = selected + [feature]

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

        except Exception:
            continue

    if best_feature is None:
        print("No valid feature found. Stopping.")
        break

    selected.append(best_feature)
    remaining.remove(best_feature)

    scores_log.append((best_feature, best_score))

    print(f"Selected: {best_feature} | Score: {round(best_score, 5)}")

# -----------------------------
# RESULT DATAFRAME
# -----------------------------
forward_df = pd.DataFrame(
    scores_log,
    columns=["Feature", "Score"]
)

print("\nFinal Forward Selected Features:\n")
print(forward_df)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

forward_df.to_csv(os.path.join(save_dir, "forward.csv"), index=False)

with open(os.path.join(save_dir, "forward_features.txt"), "w") as f:
    f.write(", ".join(forward_df["Feature"].tolist()))

print(f"\n Forward results saved in: results/{DATASET}/forward.csv")
