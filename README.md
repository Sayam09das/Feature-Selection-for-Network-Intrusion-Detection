# Feature Selection for Network Intrusion Detection

## Project Overview

This project implements and compares multiple **feature selection algorithms** to identify the most discriminative features for network intrusion detection using the **UNSW-NB15** dataset. The goal is to determine which feature selection method best reduces dimensionality while maintaining (or improving) model performance.

## Dataset

- **Dataset:** NF-UNSW-NB15-v3 (Network Flow-based format)
- **Total Records:** 2,350,609 samples
- **Features:** 55 features (after cleaning)
- **Target:** Binary classification (Normal: 0, Attack: 1)
- **Distribution:** 2,222,930 benign vs 127,679 attack samples

## Feature Selection Algorithms Compared

### 1. **Chi-Square Test** (`Feature_Selection.py`)
- **Method:** Statistical test for feature independence
- **Best For:** Categorical/discrete features
- **Top 20 Features Selected:**
  - `MIN_TTL`, `MAX_TTL`, `FLOW_END_MILLISECONDS`, `FLOW_START_MILLISECONDS`
  - `L4_DST_PORT`, `DST_TO_SRC_AVG_THROUGHPUT`, `SERVER_TCP_FLAGS`
  - And 13 more features

**Advantages:**
- Fast computation
- Statistically sound for categorical features
- Easy interpretation

---

### 2. **Dispersion Ratio** (`Feature_Selection.py`)
- **Method:** Variance-based feature selection using coefficient of variation
- **Best For:** Numerical features with high variance
- **Top 20 Features Selected:**
  - `DNS_TTL_ANSWER`, `DST_TO_SRC_AVG_THROUGHPUT`, `SRC_TO_DST_AVG_THROUGHPUT`
  - `IN_BYTES`, `FLOW_END_MILLISECONDS`, `FLOW_START_MILLISECONDS`
  - And 14 more features

**Advantages:**
- Captures feature variability
- Good for continuous features
- Computationally efficient

---

### 3. **Backward Elimination** (`backward_Elimination.py`)
- **Method:** Iterative feature removal based on p-values from logistic regression
- **Significance Level:** 0.05
- **Features Removed:** `L4_DST_PORT` (p-value: 0.987937)
- **Final Features:** 19 out of 20 chi-square features

**Top 19 Selected Features:**
- `MIN_TTL`, `MAX_TTL`, `FLOW_END_MILLISECONDS`, `FLOW_START_MILLISECONDS`
- `DST_TO_SRC_AVG_THROUGHPUT`, `SERVER_TCP_FLAGS`, `LONGEST_FLOW_PKT`, `MAX_IP_PKT_LEN`
- And 11 more features

**Advantages:**
- Considers feature interactions
- Statistically rigorous
- Iterative refinement

**Challenges:**
- ⚠️ Computational complexity with high-dimensional data
- ⚠️ Convergence issues with correlated features

---

### 4. **Random Forest Feature Importance** (`Random_Forest.py`)
- **Method:** Tree-based feature importance from ensemble learning
- **Model:** 100 decision trees
- **Sample Size:** 50,000 samples
- **Top 20 Features:** Based on information gain/impurity reduction

**Top 10 Most Important Features:**
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `SHORTEST_FLOW_PKT` | 0.2027 |
| 2 | `MIN_TTL` | 0.1756 |
| 3 | `MAX_TTL` | 0.1644 |
| 4 | `MIN_IP_PKT_LEN` | 0.1363 |
| 5 | `IPV4_SRC_ADDR` | 0.1248 |
| 6 | `SRC_TO_DST_SECOND_BYTES` | 0.0438 |
| 7 | `DST_TO_SRC_AVG_THROUGHPUT` | 0.0209 |
| 8 | `SERVER_TCP_FLAGS` | 0.0191 |
| 9 | `TCP_WIN_MAX_OUT` | 0.0172 |
| 10 | `TCP_WIN_MAX_IN` | 0.0144 |

**Advantages:**
- Handles non-linear relationships
- No convergence issues
- Captures feature interactions naturally
- Robust to outliers

---

## Algorithm Comparison

| Criterion | Chi-Square | Dispersion | Backward Elim. | Random Forest |
|-----------|-----------|-----------|---------------|---------------|
| **Speed** | ⚡⚡⚡ Fast | ⚡⚡⚡ Fast | ⚡ Slow | ⚡⚡ Medium |
| **Accuracy** | ⭐⭐⭐ Good | ⭐⭐ Fair | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Excellent |
| **Non-linear** | No | No | Limited | Yes |
| **Scalability** | Excellent | Excellent | Poor | Good |
| **Interpretability** | High | High | High | Medium |

---

## 🏆 **BEST ALGORITHM: Random Forest**

### Why Random Forest is Best:

1. **Superior Feature Ranking:** Captures non-linear relationships and feature interactions
2. **Robust Performance:** Handles the imbalanced dataset well (2222K vs 127K samples)
3. **No Convergence Issues:** Unlike Backward Elimination, works reliably with large datasets
4. **Practical Efficiency:** Balances computation time with accuracy
5. **Identified Key Features:**
   - Packet length features (`SHORTEST_FLOW_PKT`, `MIN_IP_PKT_LEN`)
   - TTL values (`MIN_TTL`, `MAX_TTL`)
   - IP addresses and throughput metrics

### Key Insights:
- **Packet size is most discriminative** (SHORTEST_FLOW_PKT: 20.27% importance)
- **TTL variations** indicate suspicious traffic patterns (35% combined importance)
- **Flow timing and network addresses** are secondary indicators
- **Throughput metrics** have lower individual importance but contribute in ensemble

---

## Project Files

```
.
├── clean_dataset.py              # Data preprocessing & cleaning
├── Feature_Selection.py           # Chi-Square & Dispersion Ratio analysis
├── backward_Elimination.py        # Logistic regression backward elimination
├── Random_Forest.py               # Tree-based feature importance
├── Cleaned_Dataset/
│   ├── NF-UNSW-NB15-v3-NO-SCALE.csv      # Unscaled data
│   └── NF-UNSW-NB15-v3-CLEANED.csv       # Scaled data
├── data/
│   └── NF-UNSW-NB15-v3.csv               # Original dataset
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install pandas numpy scikit-learn statsmodels
```

---

## Usage

### 1. Data Cleaning
```bash
"./.venv/bin/python" clean_dataset.py
```
- Handles missing values
- Removes duplicates
- Handles infinity values
- Scales numerical features

### 2. Feature Selection Comparison
```bash
"./.venv/bin/python" Feature_Selection.py
```
- Chi-Square analysis (top 20 features)
- Dispersion Ratio analysis (top 20 features)

### 3. Backward Elimination
```bash
"./.venv/bin/python" backward_Elimination.py
```
- Iterative feature removal based on p-values
- Outputs refined feature set

### 4. Random Forest Feature Importance
```bash
"./.venv/bin/python" Random_Forest.py
```
- Trains 100-tree ensemble
- Outputs feature importance scores

---

## Dependencies

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Results Summary

### Recommended Feature Set (Top 20 from Random Forest):
```python
selected_features = [
    'SHORTEST_FLOW_PKT',           # 0.2027
    'MIN_TTL',                     # 0.1756
    'MAX_TTL',                     # 0.1644
    'MIN_IP_PKT_LEN',              # 0.1363
    'IPV4_SRC_ADDR',               # 0.1248
    'SRC_TO_DST_SECOND_BYTES',     # 0.0438
    'DST_TO_SRC_AVG_THROUGHPUT',   # 0.0209
    'SERVER_TCP_FLAGS',            # 0.0191
    'TCP_WIN_MAX_OUT',             # 0.0172
    'TCP_WIN_MAX_IN',              # 0.0144
    'IPV4_DST_ADDR',               # 0.0117
    'SRC_TO_DST_AVG_THROUGHPUT',   # 0.0111
    'DST_TO_SRC_IAT_AVG',          # 0.0092
    'FLOW_DURATION_MILLISECONDS',  # 0.0075
    'DST_TO_SRC_IAT_STDDEV',       # 0.0063
    'SRC_TO_DST_IAT_AVG',          # 0.0041
    'TCP_FLAGS',                   # 0.0040
    'RETRANSMITTED_IN_PKTS',       # 0.0039
    'DST_TO_SRC_SECOND_BYTES',     # 0.0038
    'OUT_BYTES'                    # 0.0031
]
```

**Dimensionality Reduction:** 55 → 20 features (63.6% reduction)

---

## Key Findings

**Feature Selection Success:**
- Reduced features from 55 to 20 (63.6% reduction)
- Maintained discriminative power through multiple algorithms
- Identified consistent important features across methods

**Common Top Features Across All Methods:**
- TTL-related features (MIN_TTL, MAX_TTL)
- Packet size metrics (SHORTEST_FLOW_PKT, MIN_IP_PKT_LEN)
- Network flow timing

⚠️ **Challenges Encountered:**
- Data quality: Infinity and NaN values in original dataset
- Class imbalance: 17:1 ratio (benign vs attack)
- Convergence issues with Backward Elimination on high-dimensional data

---

## Future Improvements

1. Implement additional algorithms:
   - Mutual Information
   - SHAP (SHapley Additive exPlanations)
   - RFE (Recursive Feature Elimination)

2. Cross-validation with different classifiers:
   - XGBoost
   - Neural Networks
   - SVM

3. Feature engineering:
   - Polynomial features
   - Interaction terms
   - Domain-specific aggregations

4. Hyperparameter tuning for optimal feature count

---

## Author
**Final Year Project - Network Intrusion Detection**

## License
Private Project

---

## References
- UNSW-NB15 Dataset: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity-datasets/unsw-nb15-dataset/
- Scikit-learn Documentation: https://scikit-learn.org/
- Statsmodels Documentation: https://www.statsmodels.org/
