import os
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# SETTINGS
# -----------------------------
DATASETS = ["UNSW", "CICIDS", "TON"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# -----------------------------
# TARGET DETECTION
# -----------------------------
def get_target_column(df):
    for col in ["Label", "label", "Attack", "attack"]:
        if col in df.columns:
            return col
    return None

# -----------------------------
# MODEL FUNCTION
# -----------------------------
def evaluate_model(X, y, name):

    if len(y.unique()) < 2 or X.shape[1] == 0:
        return [name, 0, 0, 0, 0, 0]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel="rbf", class_weight="balanced")

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    y_pred = model.predict(X_test)

    return [
        name,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, zero_division=0),
        recall_score(y_test, y_pred, zero_division=0),
        f1_score(y_test, y_pred, zero_division=0),
        round(end - start, 4)
    ]

# -----------------------------
# MAIN LOOP
# -----------------------------
final_results = []

for DATASET in DATASETS:

    print(f"\n🚀 Processing {DATASET}...\n")

    path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

    # 🔥 LOAD FULL DATA (FIX FOR CICIDS)
    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    # -----------------------------
    # RANDOM SHUFFLE + SAMPLE (FIX)
    # -----------------------------
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sample(n=100000, random_state=42)

    # -----------------------------
    # CLEAN
    # -----------------------------
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna().reset_index(drop=True)

    # -----------------------------
    # TARGET
    # -----------------------------
    target_col = get_target_column(df)

    if target_col is None:
        print("❌ No target column → skipping")
        continue

    print(f"✅ Target column: {target_col}")

    # -----------------------------
    # REMOVE LEAKAGE
    # -----------------------------
    leak_cols = [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "DNS_QUERY_ID",
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS"
    ]

    df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

    print("After cleaning:", df.shape)

    # -----------------------------
    # CHECK DISTRIBUTION
    # -----------------------------
    print("\nClass distribution BEFORE sampling:\n", df[target_col].value_counts())

    if df[target_col].nunique() < 2:
        print("❌ Only one class → skipping dataset\n")
        continue

    # -----------------------------
    # BALANCE DATA
    # -----------------------------
    class_counts = df[target_col].value_counts()
    min_class = class_counts.min()

    balanced = []
    for cls in class_counts.index:
        balanced.append(
            df[df[target_col] == cls].sample(min_class, random_state=42)
        )

    df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced distribution:\n", df[target_col].value_counts())

    # -----------------------------
    # SPLIT
    # -----------------------------
    Y = df[target_col]
    X_full = df.drop(columns=[target_col, "Attack"], errors="ignore")

    # -----------------------------
    # SAFE FEATURES
    # -----------------------------
    def safe_features(f):
        return [col for col in f if col in X_full.columns]

    chi_features = safe_features([
        'TCP_WIN_MAX_IN','TCP_FLAGS','CLIENT_TCP_FLAGS','L7_PROTO',
        'DST_TO_SRC_IAT_MAX','DURATION_OUT','MIN_TTL','MAX_TTL',
        'PROTOCOL','DST_TO_SRC_IAT_STDDEV','L4_DST_PORT'
    ])

    dispersion_features = safe_features([
        'OUT_BYTES','IN_BYTES','DST_TO_SRC_AVG_THROUGHPUT',
        'SRC_TO_DST_AVG_THROUGHPUT','NUM_PKTS_UP_TO_128_BYTES',
        'IN_PKTS','TCP_WIN_MAX_IN','TCP_WIN_MAX_OUT','L4_DST_PORT'
    ])

    be_features = safe_features([
        'TCP_WIN_MAX_IN','TCP_FLAGS','CLIENT_TCP_FLAGS',
        'L7_PROTO','DST_TO_SRC_IAT_MAX','PROTOCOL',
        'DST_TO_SRC_IAT_STDDEV','L4_DST_PORT'
    ])

    rf_features = safe_features([
        'TCP_WIN_MAX_IN','L4_DST_PORT','IN_BYTES','OUT_BYTES',
        'LONGEST_FLOW_PKT','SERVER_TCP_FLAGS','TCP_FLAGS'
    ])

    # -----------------------------
    # RUN MODELS
    # -----------------------------
    final_results.append([DATASET] + evaluate_model(X_full[chi_features], Y, "Chi"))
    final_results.append([DATASET] + evaluate_model(X_full[dispersion_features], Y, "Disp"))
    final_results.append([DATASET] + evaluate_model(X_full[be_features], Y, "BE"))
    final_results.append([DATASET] + evaluate_model(X_full[rf_features], Y, "RF"))

# -----------------------------
# FINAL OUTPUT
# -----------------------------
columns = ["Dataset", "Method", "Accuracy", "Precision", "Recall", "F1", "Time"]

results_df = pd.DataFrame(final_results, columns=columns)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
print("\n🔥 FINAL RESULT TABLE 🔥\n")
print(results_df)

# -----------------------------
# SAVE
# -----------------------------
save_path = os.path.join(PROJECT_ROOT, "results", "final_svm_results.csv")
results_df.to_csv(save_path, index=False)

print(f"\n📁 Saved at: {save_path}")