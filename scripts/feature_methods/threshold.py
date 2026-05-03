import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

DATASET = "UNSW"   # UNSW / CICIDS / TON

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

df = pd.read_csv(data_path)
print(f"\n {DATASET} Dataset loaded!")

if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("Error: No target column found!")

leak_cols = [
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "DNS_QUERY_ID",
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"
]

df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

Y = df[target_col]
X = df.drop(columns=[target_col, "Attack", "Label"], errors="ignore")
X = X.select_dtypes(include=[np.number])
X = X.loc[:, X.nunique() > 1]

selector = VarianceThreshold(threshold=0.01)
selector.fit(X)

scores = selector.variances_

threshold_df = pd.DataFrame({
    "Feature": X.columns,
    "Score": scores
}).sort_values(by="Score", ascending=False)

print("\n Top Threshold Features:\n")
print(threshold_df.head(20))

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

threshold_df.head(20).to_csv(os.path.join(save_dir, "threshold.csv"), index=False)

print(f"\n📁 Saved: results/{DATASET}/threshold.csv")