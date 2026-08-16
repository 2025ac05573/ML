# End-to-End Classification & Model Deployment Assignment

## a. Problem Statement
The objective is to predict structural target features leveraging multi-dimensional classification methods to evaluate generalization boundaries across unique model paradigms.

## b. Dataset Description
* **Source:** Kaggle / UCI Repository
* **Instances Evaluated:** 2000 rows
* **Features Extracted:** 15 distinct structural numeric properties

## c. My GitHub Repository Link as below 
[https://github.com/2025ac05573/ML/tree/main]

## d. Models Used & Comparative Baseline

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.8120 | 0.8430 | 0.8115 | 0.8120 | 0.8117 | 0.6240 |
| Decision Tree | 0.8540 | 0.8710 | 0.8533 | 0.8540 | 0.8536 | 0.7080 |
| kNN Classifier | 0.7910 | 0.8210 | 0.7905 | 0.7910 | 0.7907 | 0.5820 |
| Naive Bayes | 0.7650 | 0.8110 | 0.7690 | 0.7650 | 0.7668 | 0.5310 |
| Random Forest | 0.8980 | 0.9420 | 0.8976 | 0.8980 | 0.8978 | 0.7960 |

### Observations About Model Performance
* **Logistic Regression:** Provides a steady baseline but underperforms on structural patterns requiring non-linear splits.
* **Decision Tree:** Separates multi-tier features efficiently but is prone to overfitting variance shifts.
* **kNN:** Struggles to handle noise ratios if local spatial neighborhoods are dense.
* **Naive Bayes:** Computes fast calculations but exhibits performance drop-offs due to internal conditional feature independence assumptions.
* **Random Forest (Ensemble):** **Overall Winner.** Successfully minimizes variant errors by leveraging bagged estimators to score high across all metrics.
