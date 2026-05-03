# Comparative Analysis of Feature Selection Methods for Efficient Machine Learning

## Abstract
This project presents a comparative study of feature selection methods for efficient machine learning on large-scale network security datasets. The work evaluates filter, wrapper, and embedded feature selection strategies to identify compact and informative feature subsets that improve classification performance while reducing dimensionality. The experimental pipeline is implemented across three benchmark datasets: `NF-UNSW-NB15-v3`, `NF-CICIDS2018-v3`, and `NF-ToN-IoT-v3`. Performance is assessed using an SVM classifier with standard evaluation metrics such as accuracy, precision, recall, F1-score, and execution time.

## Keywords
Feature Selection, Machine Learning, Comparative Analysis, Intrusion Detection, Dimensionality Reduction, SVM, Filter Methods, Wrapper Methods, Embedded Methods

## Problem Statement
Existing feature selection methods do not deliver consistently optimal performance across different datasets, which leads to unstable and dataset-dependent results. This creates several practical challenges:

- Lack of a reliable feature selection strategy that performs well across diverse data distributions
- Instability in selected features across different data splits, reducing consistency and reproducibility
- Reduced model accuracy and weak generalization in real-world deployment settings
- Need for a more efficient and dependable approach to feature selection for large and heterogeneous datasets

## Objectives
- Compare multiple feature selection methods under a unified experimental framework
- Measure the effect of each method on classification quality and computational efficiency
- Analyze how well different methods generalize across multiple benchmark datasets
- Identify methods that offer a better trade-off between accuracy, stability, and runtime

## Methodology

### 1. Filter Methods
- Information Gain
- Chi-Square Test
- Fisher Score
- Pearson's Correlation Coefficient
- Various Threshold Methods
- Mean Absolute Difference (MAD)
- Dispersion Ratio
- ANOVA

### 2. Wrapper Methods
- Forward Selection
- Backward Elimination
- Recursive Feature Elimination (RFE)

### 3. Embedded Methods
- L1 Regularization (LASSO)
- Decision Tree
- Random Forest

## Experimental Workflow
1. Raw datasets are cleaned and transformed using the preprocessing pipeline in [scripts/clean_dataset.py](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/scripts/clean_dataset.py).
2. Each feature selection method generates a ranked or reduced feature subset for a target dataset.
3. The selected features are evaluated using an SVM classifier in [scripts/SVM_comparison.py](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/scripts/SVM_comparison.py).
4. Final performance metrics are saved in [results/final_svm_results.csv](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/results/final_svm_results.csv).

## Datasets Used
- `NF-UNSW-NB15-v3`
- `NF-CICIDS2018-v3`
- `NF-ToN-IoT-v3`

Source files are stored in [data](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/data).

## Repository Structure
```text
.
├── data/
│   ├── NF-CICIDS2018-v3.csv
│   ├── NF-ToN-IoT-v3.csv
│   └── NF-UNSW-NB15-v3.csv
├── cleaned/
│   ├── CICIDS/
│   ├── TON/
│   └── UNSW/
├── results/
│   ├── CICIDS/
│   ├── TON/
│   ├── UNSW/
│   └── final_svm_results.csv
├── scripts/
│   ├── clean_dataset.py
│   ├── SVM_comparison.py
│   └── feature_methods/
│       ├── anova.py
│       ├── backward_Elimination.py
│       ├── chi_dispersion.py
│       ├── Decision_Tree.py
│       ├── Fisher.py
│       ├── forward.py
│       ├── information_gain.py
│       ├── lasso.py
│       ├── mad.py
│       ├── Pearson.py
│       ├── Random_Forest.py
│       ├── rfe.py
│       └── threshold.py
└── README.md
```

## Implementation Coverage
The current repository implementation includes the following method scripts:

- Filter: Chi-Square, Dispersion Ratio, Fisher Score, Pearson Correlation, Information Gain, MAD, ANOVA, Threshold-based selection
- Wrapper: Forward Selection, Backward Elimination, RFE
- Embedded: LASSO, Decision Tree, Random Forest

## Performance Summary
The final evaluation file shows that feature selection behavior changes across datasets, which supports the core motivation of this project: no single method is universally best for every case.

- On `UNSW`, several methods achieve near-identical top performance, including Chi-Square, Backward Elimination, Pearson, ANOVA, Information Gain, RFE, and LASSO
- On `CICIDS`, Pearson, Fisher, and ANOVA produce the highest observed accuracy in the current results
- On `TON`, Random Forest achieves the strongest overall performance among the evaluated methods

This indicates that embedded methods, especially tree-based approaches, are strong candidates for robust cross-dataset performance, while selected filter and wrapper methods remain competitive depending on data characteristics.

## How to Run

### 1. Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install required packages
```bash
pip install pandas numpy scikit-learn statsmodels
```

### 3. Clean a dataset
Update the dataset settings inside [scripts/clean_dataset.py](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/scripts/clean_dataset.py:8), then run:

```bash
python scripts/clean_dataset.py
```

### 4. Run feature selection methods
Choose the dataset name inside each script under [scripts/feature_methods](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/scripts/feature_methods), then execute the required method files.

Example:
```bash
python scripts/feature_methods/information_gain.py
python scripts/feature_methods/Pearson.py
python scripts/feature_methods/Random_Forest.py
```

### 5. Compare all methods using SVM
```bash
python scripts/SVM_comparison.py
```

## Evaluation Metrics
The study uses the following evaluation measures:

- Accuracy
- Precision
- Recall
- F1-score
- Training time

## Key Contribution
This project provides a structured comparison of classical and modern feature selection methods for efficient machine learning. Its main contribution is the cross-dataset evaluation of multiple approaches under a common classification framework, helping identify methods that are more stable, accurate, and computationally practical for real-world applications.

## Team Members
- Sayam Das
  GitHub: `sayam09das`
- Sayandeep Patra
  GitHub: `Sayandeep07`
- Sherya Sikder
  GitHub: `Sheryasikder123`
- Tanish Das
  GitHub: `Tanish-Das`

## License
This project is released under the MIT License. See [LICENSE](/Users/sayamdas/Documents/Programming/Final%20Year%20Project/LICENSE) for details.

## Conclusion
The experimental results show that feature selection remains a dataset-sensitive problem. Filter methods offer speed and interpretability, wrapper methods provide targeted refinement, and embedded methods often deliver stronger practical performance. The overall findings reinforce the need for efficient and reliable feature selection strategies that can adapt to diverse datasets while preserving model accuracy and generalization.
