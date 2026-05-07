import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, f_classif

DATASET = "UNSW"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

df = pd.read_csv(data_path)

print(f"\n {DATASET} Dataset loaded!")

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

k = 20

selector = SelectKBest(score_func=f_classif, k=k)
selector.fit(X, Y)

scores = selector.scores_

anova_df = pd.DataFrame({
    "Feature": X.columns,
    "Score": scores
})

anova_df = anova_df.sort_values(by="Score", ascending=False)

anova_top = anova_df.head(k)

print("\nTop Features using ANOVA:\n")
print(anova_top)

anova_features = anova_top["Feature"].values

print("\nSelected ANOVA Features:\n")
print(anova_features)

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

anova_top.to_csv(os.path.join(save_dir, "anova.csv"), index=False)

with open(os.path.join(save_dir, "feature_lists_anova.txt"), "w") as f:
    f.write(", ".join(anova_features))

print(f"\nANOVA results saved in: results/{DATASET}/")
