import numpy as np
import pandas as pd
import os

DATASET = "UNSW"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

df = pd.read_csv(data_path)

print(f"\n{DATASET} Dataset loaded!")

if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("No target column found!")

Y = df[target_col]

drop_cols = [col for col in ["Label", "Attack"] if col in df.columns]
X = df.drop(columns=drop_cols)

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

X = X.loc[:, X.nunique() > 1]

print(f"\nAfter cleaning: {X.shape}")

mad_scores = {}

for col in X.columns:
    mad_scores[col] = np.mean(np.abs(X[col] - np.mean(X[col])))

mad_df = pd.DataFrame({
    "Feature": mad_scores.keys(),
    "Score": mad_scores.values()
})

mad_df = mad_df.sort_values(by="Score", ascending=False)

k = 20
mad_top = mad_df.head(k)

print("\nTop Features using MAD:\n")
print(mad_top)

mad_features = mad_top["Feature"].values

print("\nSelected MAD Features:\n")
print(mad_features)

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

mad_top.to_csv(os.path.join(save_dir, "mad.csv"), index=False)

with open(os.path.join(save_dir, "feature_lists_mad.txt"), "w") as f:
    f.write(", ".join(mad_features))

print(f"\nMAD results saved in: results/{DATASET}/")
