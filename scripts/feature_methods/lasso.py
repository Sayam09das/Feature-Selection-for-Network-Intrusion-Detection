import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DATASET = "CICIDS"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

df = pd.read_csv(data_path)
print(f"\n{DATASET} Dataset loaded!")

if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("No target column found!")

leak_cols = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "DNS_QUERY_ID",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
]

df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")
df = df.sample(n=min(30000, len(df)), random_state=42)
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

Y = df[target_col]
X = df.drop(columns=[target_col, "Attack", "Label"], errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.loc[:, X.nunique() > 1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(
    penalty="l1",
    solver="saga",
    max_iter=5000,
    random_state=42
)

model.fit(X_scaled, Y)

coef = np.abs(model.coef_).mean(axis=0)

lasso_df = pd.DataFrame({
    "Feature": X.columns,
    "Score": coef
})

lasso_df = lasso_df[lasso_df["Score"] > 0]
lasso_df = lasso_df.sort_values(by="Score", ascending=False)

print("\nTop LASSO Features:\n")
print(lasso_df.head(20))

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

lasso_df.head(20).to_csv(os.path.join(save_dir, "lasso.csv"), index=False)

print(f"\nSaved: results/{DATASET}/lasso.csv")
