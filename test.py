import time
import numpy as np
import pandas as pd
from google.colab import files

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================= LOAD DATA =================
print("Please upload the file: NF-UNSW-NB15-v3-CLEANED.csv")
uploaded = files.upload()
file_name = list(uploaded.keys())[0]

df = pd.read_csv(file_name)

# Drop unwanted columns
df_selection = df.drop(columns=['FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'Label'])

# Numerical features
numerical_features = df_selection.select_dtypes(include=['number']).columns.tolist()

if 'Attack' in numerical_features:
    numerical_features.remove('Attack')

# 🔴 Remove rare classes
class_counts = df_selection['Attack'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df_filtered = df_selection[df_selection['Attack'].isin(valid_classes)]

# Define X and y
X = df_filtered[numerical_features]
y = df_filtered['Attack']

# Handle NaN and infinity
X = X.fillna(0)
X = X.replace([np.inf, -np.inf], 0)

# Remove constant features before splitting to ensure valid feature selection
# This addresses the UserWarning: Features are constant.
constant_features = X.columns[X.nunique() == 1]
if not constant_features.empty:
    print(f"\nRemoving constant features: {list(constant_features)}")
    X = X.drop(columns=constant_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= RANDOM FOREST FUNCTION =================
def run_rf(selected_features, name):
    start = time.time()

    X_tr = X_train[selected_features]
    X_te = X_test[selected_features]

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)

    end = time.time()

    print(f"\n{name} Results:")
    print("Selected Features:", selected_features)

    print("\nAccuracy:", acc)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0)) # Set zero_division to 0

    print("Execution Time:", round(end - start, 4), "seconds")


# ================= 1. MAD =================
mad_scores = {}

for col in X.columns:
    mad_scores[col] = np.mean(np.abs(X[col] - np.mean(X[col])))

mad_df = pd.DataFrame(mad_scores.items(), columns=['Feature', 'MAD'])
mad_features = mad_df.sort_values(by='MAD', ascending=False).head(5)['Feature'].tolist()

run_rf(mad_features, "MAD")


# ================= 2. ANOVA =================
anova_selector = SelectKBest(score_func=f_classif, k=5)
anova_selector.fit(X, y) # X here is already cleaned of constant features

anova_features = X.columns[anova_selector.get_support()].tolist()

run_rf(anova_features, "ANOVA")


# ================= 3. FORWARD SELECTION =================
model = LogisticRegression(max_iter=1000, solver='liblinear')

selected = []
remaining = list(X.columns)

for i in range(5):
    best_score = -1
    best_feature = None

    for feature in remaining:
        temp = selected + [feature]
        scores = cross_val_score(model, X[temp], y, cv=2)
        score = np.mean(scores)

        if score > best_score:
            best_score = score
            best_feature = feature

    selected.append(best_feature)
    remaining.remove(best_feature)

forward_features = selected

run_rf(forward_features, "Forward Selection") this my another friend code mad,anvova,foward selection ok smae type 1st make all three files then code give me 