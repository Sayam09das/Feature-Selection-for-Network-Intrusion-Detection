import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler

DATASET = "UNSW"   # UNSW / CICIDS / TON

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

df = pd.read_csv(data_path)
print(f"\n✅ {DATASET} Dataset loaded!")

if "Label" in df.columns:
    target_col = "Label"
elif "Attack" in df.columns:
    target_col = "Attack"
else:
    raise Exception("❌ No target column found!")

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

model = LogisticRegression(max_iter=5000, solver="liblinear")
selector = RFE(model, n_features_to_select=20)

selector.fit(X_scaled, Y)

rfe_df = pd.DataFrame({
    "Feature": X.columns,
    "Score": selector.ranking_
}).sort_values(by="Score", ascending=True)

top_features = rfe_df[rfe_df["Score"] == 1]

print("\n🔥 Top RFE Features:\n")
print(top_features)

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

top_features.to_csv(os.path.join(save_dir, "rfe.csv"), index=False)

print(f"\n📁 Saved: results/{DATASET}/rfe.csv")