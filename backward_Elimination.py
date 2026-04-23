import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Load NON-SCALED dataset
# -----------------------------
df = pd.read_csv("Cleaned_Dataset/NF-UNSW-NB15-v3-NO-SCALE.csv")

print("Dataset loaded!")

# Handle inf and NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean(numeric_only=True))

# Remove rows with remaining NaN values
df = df.dropna()

print(f"Dataset cleaned. Shape: {df.shape}")

# Scale features for numerical stability
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(["Label", "Attack"], errors='ignore')
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Use smaller sample for numerical stability
df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)

# Define X and Y
Y = df_sample["Label"]

chi_features = [
    'MIN_TTL','MAX_TTL','FLOW_END_MILLISECONDS','FLOW_START_MILLISECONDS',
    'L4_DST_PORT','DST_TO_SRC_AVG_THROUGHPUT','SERVER_TCP_FLAGS',
    'LONGEST_FLOW_PKT','MAX_IP_PKT_LEN','IPV4_DST_ADDR',
    'SRC_TO_DST_SECOND_BYTES','IPV4_SRC_ADDR','TCP_FLAGS',
    'CLIENT_TCP_FLAGS','DNS_QUERY_ID','FTP_COMMAND_RET_CODE',
    'MIN_IP_PKT_LEN','TCP_WIN_MAX_IN','SHORTEST_FLOW_PKT','DST_TO_SRC_IAT_AVG'
]

X = df_sample[chi_features]

# Scale features for stability
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=chi_features)

# 🔥 ALIGN INDICES
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

# Add constant
X = sm.add_constant(X)

# Backward Elimination with error handling
significance_level = 0.05
max_iterations = 20
iteration = 0

while iteration < max_iterations:
    iteration += 1
    try:
        # Fit logistic regression with options for stability
        model = sm.Logit(Y, X).fit(disp=0, maxiter=1000)
        p_values = model.pvalues
        
        max_p = p_values.max()
        
        if max_p > significance_level:
            feature_to_remove = p_values.idxmax()
            if feature_to_remove != "const":
                print(f"Iteration {iteration}: Removing {feature_to_remove} (p-value: {max_p:.6f})")
                X = X.drop(columns=[feature_to_remove])
            else:
                print("Backward elimination complete!")
                break
        else:
            print("Backward elimination complete!")
            break
    except Exception as e:
        print(f"Iteration {iteration}: Model fitting error - {str(e)}")
        print("Backward elimination stopped due to convergence issues.")
        break

# -----------------------------
# Final Selected Features
# -----------------------------
selected_features = X.columns

print("\nFinal Selected Features (Backward Elimination):\n")
print(selected_features)