import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
from sklearn.preprocessing import StandardScaler

# -----------------------------
# SETTINGS
# -----------------------------
DATASET = "UNSW"   # UNSW / CICIDS / TON

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

data_path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/no_scale.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(data_path)

print(f"\nDataset loaded: {DATASET}")

# -----------------------------
# CLEANING
# -----------------------------
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))
df = df.dropna()

print(f"Dataset cleaned. Shape: {df.shape}")

# -----------------------------
# SAMPLING (IMPORTANT)
# -----------------------------
df = df.sample(n=30000, random_state=42).reset_index(drop=True)

# -----------------------------
# TARGET
# -----------------------------
Y = df["Label"]

# -----------------------------
# FEATURE SET (TON CHI-SQUARE)
# -----------------------------
# chi_features = [
#     'TCP_WIN_MAX_IN','TCP_FLAGS','CLIENT_TCP_FLAGS',
#     'DNS_QUERY_ID','L7_PROTO','DST_TO_SRC_IAT_MAX',
#     'DURATION_OUT','MIN_TTL','MAX_TTL','PROTOCOL',
#     'DST_TO_SRC_IAT_STDDEV','FLOW_START_MILLISECONDS',
#     'FLOW_END_MILLISECONDS','L4_DST_PORT','FLOW_DURATION_MILLISECONDS',
#     'DURATION_IN','ICMP_TYPE','ICMP_IPV4_TYPE',
#     'DST_TO_SRC_SECOND_BYTES','DST_TO_SRC_IAT_AVG'
# ]


# ------------------------------
# FEATURE SET (CICIDS CHI-SQUARE)
# ------------------------------
# chi_features = [
#     'TCP_WIN_MAX_IN','TCP_FLAGS','CLIENT_TCP_FLAGS',
#     'L7_PROTO','DST_TO_SRC_IAT_MAX','DURATION_OUT',
#     'MIN_TTL','MAX_TTL','PROTOCOL',
#     'DST_TO_SRC_IAT_STDDEV','L4_DST_PORT',
#     'FLOW_DURATION_MILLISECONDS','DURATION_IN',
#     'ICMP_TYPE','ICMP_IPV4_TYPE',
#     'DST_TO_SRC_SECOND_BYTES','DST_TO_SRC_IAT_AVG'
# ]

# -----------------------------
# FEATURE SET (UNSW CHI-SQUARE)
# -----------------------------
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

# -----------------------------
# REMOVE CONSTANT FEATURES
# -----------------------------
X = X.loc[:, X.nunique() > 1]

# -----------------------------
# REMOVE HIGH CORRELATION
# -----------------------------
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X = X.drop(columns=to_drop)

print("Removed correlated features:", to_drop)

# -----------------------------
# SCALE (important)
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=[c for c in chi_features if c not in to_drop])

# -----------------------------
# ALIGN INDEX (CRITICAL FIX)
# -----------------------------
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

# -----------------------------
# ADD CONSTANT
# -----------------------------
X = sm.add_constant(X)

# -----------------------------
# BACKWARD ELIMINATION
# -----------------------------
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

# -----------------------------
# FINAL FEATURES
# -----------------------------
selected_features = [col for col in X.columns if col != "const"]

print("\nFinal Selected Features (Backward Elimination):\n")
print(selected_features)

# -----------------------------
# SAVE RESULTS
# -----------------------------
save_dir = os.path.join(PROJECT_ROOT, f"results/{DATASET}")
os.makedirs(save_dir, exist_ok=True)

# -----------------------------
# SAVE RESULTS (FIXED)
# -----------------------------
be_df = pd.DataFrame({
    "Feature": selected_features,
    "Score": [1]*len(selected_features)
})

be_df.to_csv(os.path.join(save_dir, "be.csv"), index=False)

print(f"\nBE features saved in: results/{DATASET}/be.csv")