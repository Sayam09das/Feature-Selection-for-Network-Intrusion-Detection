import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.preprocessing import StandardScaler

DATASET = "UNSW"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

df = pd.read_csv(data_path)

print(f"\nDataset loaded: {DATASET}")

df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))
df = df.dropna()

print(f"Dataset cleaned. Shape: {df.shape}")

df = df.sample(n=30000, random_state=42).reset_index(drop=True)

Y = df["Label"]

chi_features = [
    'MIN_TTL','MAX_TTL','FLOW_END_MILLISECONDS','FLOW_START_MILLISECONDS',
    'L4_DST_PORT','DST_TO_SRC_AVG_THROUGHPUT','SERVER_TCP_FLAGS',
    'LONGEST_FLOW_PKT','MAX_IP_PKT_LEN','IPV4_DST_ADDR',
    'SRC_TO_DST_SECOND_BYTES','IPV4_SRC_ADDR','TCP_FLAGS',
    'CLIENT_TCP_FLAGS','DNS_QUERY_ID','FTP_COMMAND_RET_CODE',
    'MIN_IP_PKT_LEN','TCP_WIN_MAX_IN','SHORTEST_FLOW_PKT',
    'DST_TO_SRC_IAT_AVG'
]

X = df[chi_features]

X = X.loc[:, X.nunique() > 1]

corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X.drop(columns=to_drop)

print("Removed correlated features:", to_drop)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=[c for c in chi_features if c not in to_drop])

X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

X = sm.add_constant(X)

significance_level = 0.05

while True:
    try:
        model = sm.Logit(Y, X).fit(disp=0)
        p_values = model.pvalues

        max_p = p_values.max()

        if max_p > significance_level:
            feature_to_remove = p_values.idxmax()

            if feature_to_remove == "const":
                break

            print(f"Removing {feature_to_remove} (p-value: {max_p:.6f})")
            X = X.drop(columns=[feature_to_remove])
        else:
            break

    except Exception as e:
        print("Model error:", e)
        print("Stopping due to instability.")
        break

selected_features = [col for col in X.columns if col != "const"]

print("\nFinal Selected Features (Backward Elimination):\n")
print(selected_features)

save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

be_df = pd.DataFrame({
    "Feature": selected_features,
    "Score": [1]*len(selected_features)
})

be_df.to_csv(os.path.join(save_dir, "be.csv"), index=False)

print(f"\nBE features saved in: results/{DATASET}/be.csv")
