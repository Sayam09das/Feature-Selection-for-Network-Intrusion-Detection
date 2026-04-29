import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load datasets
# -----------------------------
df_scaled = pd.read_csv("Cleaned_Dataset/NF-UNSW-NB15-v3-CLEANED.csv")       # for Chi-square
df_original = pd.read_csv("Cleaned_Dataset/NF-UNSW-NB15-v3-NO-SCALE.csv")    # for Dispersion

print("Datasets loaded!")

# -----------------------------
# Target & Features
# -----------------------------
Y = df_scaled["Label"]

X_scaled = df_scaled.drop(["Label", "Attack"], axis=1)
X_original = df_original.drop(["Label", "Attack"], axis=1)

print("\nTarget distribution:\n")
print(Y.value_counts())

# -----------------------------
# Scale for Chi-square (0-1 range)
# -----------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_scaled)
X_scaled = pd.DataFrame(X_scaled, columns=X_original.columns)

# =============================
# CHI-SQUARE
# =============================
k = 20

selector = SelectKBest(score_func=chi2, k=k)
selector.fit(X_scaled, Y)

scores = selector.scores_

feature_scores = pd.DataFrame({
    "Feature": X_original.columns,
    "Score": scores
})

feature_scores = feature_scores.sort_values(by="Score", ascending=False)

print("\nTop Features using Chi-Square:\n")
print(feature_scores.head(k))

chi_features = feature_scores.head(k)["Feature"].values

print("\nSelected Chi-Square Features:\n")
print(chi_features)

# =============================
# DISPERSION RATIO
# =============================
mean = X_original.mean()
variance = X_original.var()

dispersion_ratio = variance / (mean + 1e-5)

dispersion_df = pd.DataFrame({
    "Feature": X_original.columns,
    "Dispersion": dispersion_ratio
})

dispersion_df = dispersion_df.sort_values(by="Dispersion", ascending=False)

print("\nTop Features using Dispersion Ratio:\n")
print(dispersion_df.head(k))

dispersion_features = dispersion_df.head(k)["Feature"].values

print("\nSelected Dispersion Features:\n")
print(dispersion_features)