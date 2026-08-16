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
| Logistic Regression | 0.975 | 0.9996 | 0.9751 | 0.9747 | 0.9744 | 0.9669 |
| Decision Tree | 0.7975 | 0.9329 | 0.7982 | 0.7942 | 0.7949 | 0.7307 |
| kNN Classifier | 0.53 | 0.7629 | 0.5514 | 0.5221 | 0.5282 | 0.3789 |
| Naive Bayes | 0.7975 | 0.956 | 0.7983 | 0.7926 | 0.7929 | 0.7313 |
| Random Forest | 0.8925 | 0.9831 | 0.8916 | 0.8914 | 0.8905 | 0.8572 |

### Observations About Model Performance
* **Logistic Regression:** Provides a steady baseline but underperforms on structural patterns requiring non-linear splits.
* **Decision Tree:** Separates multi-tier features efficiently but is prone to overfitting variance shifts.
* **kNN:** Struggles to handle noise ratios if local spatial neighborhoods are dense.
* **Naive Bayes:** Computes fast calculations but exhibits performance drop-offs due to internal conditional feature independence assumptions.
* **Random Forest (Ensemble):** **Overall Winner.** Successfully minimizes variant errors by leveraging bagged estimators to score high across all metrics.
