import pandas as pd
import os
from sklearn.tree import DecisionTreeClassifier

DATASET = "CICIDS"

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

print("\nTarget distribution:\n")
print(Y.value_counts())

import numpy as np
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X, Y)

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

k = 20
top_features = feature_importance.head(k)

print("\nTop Features using Decision Tree:\n")
print(top_features)

dt_features = top_features["Feature"].values

print("\nSelected Decision Tree Features:\n")
print(dt_features)

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

top_features.to_csv(
    os.path.join(save_dir, "decision_tree.csv"),
    index=False
)

with open(os.path.join(save_dir, "feature_lists_dt.txt"), "w") as f:
    f.write("Decision Tree:\n" + ", ".join(dt_features))

print(f"\nResults saved in: results/{DATASET}/")
