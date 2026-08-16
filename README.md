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
### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | **Overall Winner for this dataset.** Achieved the highest accuracy (97.5%) and near-perfect AUC (0.9996). The feature relationships in the Mobile Price dataset are highly linear, allowing it to capture target boundaries exceptionally well. |
| **Decision Tree** | Separates multi-tier features efficiently but is prone to overfitting variance shifts, leading to a moderate performance baseline (79.75% accuracy). |
| **kNN** | Struggles to handle noise ratios if local spatial neighborhoods are dense, resulting in poor performance with an accuracy of only 53.00%. |
| **Naive Bayes** | Computes fast calculations but exhibits performance drop-offs (79.75% accuracy) due to internal conditional feature independence assumptions. |
| **Random Forest (Ensemble)** | Strong performance (89.25% accuracy). Successfully minimizes variant errors by leveraging bagged estimators to score high across metrics, though it slightly over-complicates the clean linear patterns captured better by Logistic Regression. |

#### Overall Winner for your dataset?
**Logistic Regression** is the clear overall winner for this dataset. It outperformed all other models across every single metric, including Accuracy (0.9750), AUC (0.9996), and MCC (0.9669). This demonstrates that the engineered features for mobile phone pricing share a highly predictable linear relationship with the target pricing tiers.

