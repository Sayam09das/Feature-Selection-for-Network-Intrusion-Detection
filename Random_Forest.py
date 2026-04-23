import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load dataset (NO SCALE needed)
# -----------------------------
df = pd.read_csv("Cleaned_Dataset/NF-UNSW-NB15-v3-NO-SCALE.csv")

print("Dataset loaded!")

# Handle inf and NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(f"Data cleaned. Shape: {df.shape}")

# Sampling (important for speed)
# ------------------------------ 
df_sample = df.sample(n=50000, random_state=42).reset_index(drop=True)

# Define X and Y
# ------------------------------ 
Y = df_sample["Label"]
X = df_sample.drop(["Label", "Attack"], axis=1)

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X, Y)

# -----------------------------
# Get Feature Importance
# -----------------------------
importance = rf.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

# Sort features
feature_importance_df = feature_importance_df.sort_values(
    by="Importance", ascending=False
)

# -----------------------------
# Show Top Features
# -----------------------------
k = 20

print("\nTop Features using Random Forest:\n")
print(feature_importance_df.head(k))

# -----------------------------
# Selected Features
# -----------------------------
rf_features = feature_importance_df.head(k)["Feature"].values

print("\nSelected Random Forest Features:\n")
print(rf_features)