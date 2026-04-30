import os
import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -----------------------------
# Fix display issue (SHOW ALL COLUMNS)
# -----------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# -----------------------------
# Load Dataset (FAST)
# -----------------------------
df = pd.read_csv(
    "cleaned/UNSW/NF-UNSW-NB15-v3-CLEANED.csv",
    nrows=100000   # load partial for speed
)

print("Dataset loaded!")

# -----------------------------
# Sampling (important)
# -----------------------------
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

Y = df["Label"]

# -----------------------------
# Feature Sets
# -----------------------------
chi_features = [
    'MIN_TTL','MAX_TTL','FLOW_END_MILLISECONDS','FLOW_START_MILLISECONDS',
    'L4_DST_PORT','DST_TO_SRC_AVG_THROUGHPUT','SERVER_TCP_FLAGS',
    'LONGEST_FLOW_PKT','MAX_IP_PKT_LEN','IPV4_DST_ADDR',
    'SRC_TO_DST_SECOND_BYTES','IPV4_SRC_ADDR','TCP_FLAGS',
    'CLIENT_TCP_FLAGS','DNS_QUERY_ID','FTP_COMMAND_RET_CODE',
    'MIN_IP_PKT_LEN','TCP_WIN_MAX_IN','SHORTEST_FLOW_PKT','DST_TO_SRC_IAT_AVG'
]

dispersion_features = [
    'DNS_TTL_ANSWER','DST_TO_SRC_AVG_THROUGHPUT','SRC_TO_DST_AVG_THROUGHPUT',
    'IN_BYTES','FLOW_END_MILLISECONDS','FLOW_START_MILLISECONDS',
    'RETRANSMITTED_IN_BYTES','OUT_BYTES','RETRANSMITTED_OUT_BYTES',
    'DNS_QUERY_ID','L4_DST_PORT','DURATION_IN','FLOW_DURATION_MILLISECONDS',
    'ICMP_TYPE','TCP_WIN_MAX_IN','L4_SRC_PORT','SRC_TO_DST_IAT_MIN',
    'TCP_WIN_MAX_OUT','DURATION_OUT','DNS_QUERY_TYPE'
]

be_features = [
    'MIN_TTL','MAX_TTL','FLOW_END_MILLISECONDS','FLOW_START_MILLISECONDS',
    'DST_TO_SRC_AVG_THROUGHPUT','SERVER_TCP_FLAGS','LONGEST_FLOW_PKT',
    'MAX_IP_PKT_LEN','IPV4_DST_ADDR','SRC_TO_DST_SECOND_BYTES',
    'IPV4_SRC_ADDR','TCP_FLAGS','CLIENT_TCP_FLAGS','DNS_QUERY_ID',
    'FTP_COMMAND_RET_CODE','MIN_IP_PKT_LEN','TCP_WIN_MAX_IN',
    'SHORTEST_FLOW_PKT','DST_TO_SRC_IAT_AVG'
]

rf_features = [
    'SHORTEST_FLOW_PKT','MIN_TTL','MAX_TTL','MIN_IP_PKT_LEN',
    'IPV4_SRC_ADDR','SRC_TO_DST_SECOND_BYTES','DST_TO_SRC_AVG_THROUGHPUT',
    'SERVER_TCP_FLAGS','TCP_WIN_MAX_OUT','TCP_WIN_MAX_IN',
    'IPV4_DST_ADDR','SRC_TO_DST_AVG_THROUGHPUT','DST_TO_SRC_IAT_AVG',
    'FLOW_DURATION_MILLISECONDS','DST_TO_SRC_IAT_STDDEV',
    'SRC_TO_DST_IAT_AVG','TCP_FLAGS','RETRANSMITTED_IN_PKTS',
    'DST_TO_SRC_SECOND_BYTES','OUT_BYTES'
]

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(X, Y, name):

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Scale (important for SVM)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM Model (fast version)
    model = SVC(kernel='linear')

    # Time tracking
    start_time = time.time()
    model.fit(X_train, Y_train)
    end_time = time.time()

    Y_pred = model.predict(X_test)

    return {
        "Method": name,
        "Accuracy": accuracy_score(Y_test, Y_pred),
        "Precision": precision_score(Y_test, Y_pred, zero_division=0),
        "Recall": recall_score(Y_test, Y_pred, zero_division=0),
        "F1-Score": f1_score(Y_test, Y_pred, zero_division=0),
        "Time (s)": round(end_time - start_time, 4)
    }

# -----------------------------
# Run All Methods
# -----------------------------
results = []

results.append(evaluate_model(df[chi_features], Y, "Chi-Square"))
results.append(evaluate_model(df[dispersion_features], Y, "Dispersion"))
results.append(evaluate_model(df[be_features], Y, "Backward Elimination"))
results.append(evaluate_model(df[rf_features], Y, "Random Forest"))

# -----------------------------
# Final Results
# -----------------------------
results_df = pd.DataFrame(results)

# -----------------------------
# Save Results to Files
# -----------------------------
PROJECT_ROOT = "/Users/sayamdas/Documents/Programming/Final Year Project"
DATASET = "UNSW"

save_dir = os.path.join(PROJECT_ROOT, "results", DATASET)
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "svm_results.csv")
results_df.to_csv(csv_path, index=False)

txt_path = os.path.join(save_dir, "selected_features.txt")
with open(txt_path, "w") as f:
    f.write("Chi-Square Features:\n")
    f.write(", ".join(chi_features) + "\n\n")
    f.write("Dispersion Features:\n")
    f.write(", ".join(dispersion_features) + "\n\n")
    f.write("Backward Elimination Features:\n")
    f.write(", ".join(be_features) + "\n\n")
    f.write("Random Forest Features:\n")
    f.write(", ".join(rf_features) + "\n")

print("\n===== FINAL SVM COMPARISON =====\n")
print(results_df.to_string(index=False))
