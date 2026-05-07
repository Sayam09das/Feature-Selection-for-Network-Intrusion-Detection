import os
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATASETS = ["UNSW", "CICIDS", "TON"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

def get_target_column(df):
    for col in ["Label", "label", "Attack", "attack"]:
        if col in df.columns:
            return col
    return None

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

final_results = []

for DATASET in DATASETS:

    print(f"\nProcessing {DATASET}...\n")

    path = os.path.join(PROJECT_ROOT, f"cleaned/{DATASET}/cleaned.csv")

    df = pd.read_csv(path)

    print("Original shape:", df.shape)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sample(n=100000, random_state=42)

    df = df.replace([float("inf"), -float("inf")], pd.NA)
    df = df.dropna().reset_index(drop=True)

    target_col = get_target_column(df)

    if target_col is None:
        print("No target column → skipping")
        continue

    print(f"Target column: {target_col}")

    leak_cols = [
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "DNS_QUERY_ID",
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS"
    ]

    df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors="ignore")

    print("After cleaning:", df.shape)

    print("\nClass distribution BEFORE sampling:\n", df[target_col].value_counts())
    if df[target_col].nunique() < 2:
        print("Only one class → skipping dataset\n")
        continue

    class_counts = df[target_col].value_counts()
    min_class = class_counts.min()

    balanced = []
    for cls in class_counts.index:
        balanced.append(
            df[df[target_col] == cls].sample(min_class, random_state=42)
        )

    df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced distribution:\n", df[target_col].value_counts())

    Y = df[target_col]
    X_full = df.drop(columns=[target_col, "Attack"], errors="ignore")

    def load_features(method_name):
        path = os.path.join(PROJECT_ROOT, f"results/{DATASET}/{method_name}.csv")
        if not os.path.exists(path):
            print(f"⚠️ Missing {method_name} features")
            return []
        df_feat = pd.read_csv(path)
        features = [f for f in df_feat["Feature"].head(20).tolist() if f in X_full.columns]
        if len(features) == 0:
            print(f"⚠️ {method_name} has 0 valid features")
        return features

    chi_features = load_features("chi_square")
    dispersion_features = load_features("dispersion")
    be_features = load_features("be")
    rf_features = load_features("random_forest")
    pearson_features = load_features("pearson")
    fisher_features = load_features("fisher")
    dt_features = load_features("decision_tree")
    mad_features = load_features("mad")
    anova_features = load_features("anova")
    forward_features = load_features("forward")
    ig_features = load_features("information_gain")
    threshold_features = load_features("threshold")
    rfe_features = load_features("rfe")
    lasso_features = load_features("lasso")

    final_results.append([DATASET] + evaluate_model(X_full[chi_features], Y, "Chi"))
    final_results.append([DATASET] + evaluate_model(X_full[dispersion_features], Y, "Disp"))
    final_results.append([DATASET] + evaluate_model(X_full[be_features], Y, "BE"))
    final_results.append([DATASET] + evaluate_model(X_full[rf_features], Y, "RF"))
    final_results.append([DATASET] + evaluate_model(X_full[pearson_features], Y, "Pearson"))
    final_results.append([DATASET] + evaluate_model(X_full[fisher_features], Y, "Fisher"))
    final_results.append([DATASET] + evaluate_model(X_full[dt_features], Y, "DecisionTree"))
    final_results.append([DATASET] + evaluate_model(X_full[mad_features], Y, "MAD"))
    final_results.append([DATASET] + evaluate_model(X_full[anova_features], Y, "ANOVA"))
    final_results.append([DATASET] + evaluate_model(X_full[forward_features], Y, "Forward"))
    final_results.append([DATASET] + evaluate_model(X_full[ig_features], Y, "InformationGain"))
    final_results.append([DATASET] + evaluate_model(X_full[threshold_features], Y, "Threshold"))
    final_results.append([DATASET] + evaluate_model(X_full[rfe_features], Y, "RFE"))
    final_results.append([DATASET] + evaluate_model(X_full[lasso_features], Y, "LASSO"))

columns = ["Dataset", "Method", "Accuracy", "Precision", "Recall", "F1", "Time"]

results_df = pd.DataFrame(final_results, columns=columns)

filter_methods = [
    "Chi",
    "Disp",
    "Pearson",
    "Fisher",
    "MAD",
    "ANOVA",
    "InformationGain",
    "Threshold"
]

wrapper_methods = [
    "BE",
    "Forward",
    "RFE"
]

embedded_methods = [
    "RF",
    "DecisionTree",
    "LASSO"
]

filter_df = results_df[results_df["Method"].isin(filter_methods)]
wrapper_df = results_df[results_df["Method"].isin(wrapper_methods)]
embedded_df = results_df[results_df["Method"].isin(embedded_methods)]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
print("\nFINAL RESULT TABLE\n")
print(results_df)
print("\n========== FILTER METHODS RESULT TABLE ==========\n")
print(filter_df)
print("\n========== WRAPPER METHODS RESULT TABLE ==========\n")
print(wrapper_df)
print("\n========== EMBEDDED METHODS RESULT TABLE ==========\n")
print(embedded_df)

filter_df.to_csv(os.path.join(PROJECT_ROOT, "results", "filter_methods_results.csv"), index=False)
wrapper_df.to_csv(os.path.join(PROJECT_ROOT, "results", "wrapper_methods_results.csv"), index=False)
embedded_df.to_csv(os.path.join(PROJECT_ROOT, "results", "embedded_methods_results.csv"), index=False)

print("\nSaved separate tables:")
print("Filter   -> results/filter_methods_results.csv")
print("Wrapper  -> results/wrapper_methods_results.csv")
print("Embedded -> results/embedded_methods_results.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1", "Time"]

for metric in metrics:
    plt.figure(figsize=(16, 7))

    sns.barplot(
        data=results_df,
        x="Method",
        y=metric,
        hue="Dataset"
    )

    plt.title(f"{metric} Comparison Across Datasets", fontsize=16)
    plt.xlabel("Method")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")

    if metric != "Time":
        plt.ylim(0.90, 1.01)

    plt.legend(title="Dataset")
    plt.tight_layout()

    plot_path = os.path.join(
        PROJECT_ROOT,
        "results",
        f"{metric.lower()}_comparison.png"
    )

    plt.savefig(plot_path, dpi=300)
    plt.show()
    plt.close()

    print(f"{metric} comparison plot saved at: {plot_path}")

save_path = os.path.join(PROJECT_ROOT, "results", "final_svm_results.csv")
results_df.to_csv(save_path, index=False)

print(f"\nSaved at: {save_path}")
