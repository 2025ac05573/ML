import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1. Create or load your dataset (Simulated here with exact shape requirement)
np.random.seed(42)
X = np.random.randn(2000, 15)  # 2000 rows, 15 features (meets >500 rows, >12 features)
y = np.random.randint(0, 2, size=2000) # Binary classification

feature_names = [f"feature_{i}" for i in range(15)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], df['target'], test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save test dataset for Streamlit upload requirement
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test
test_df.to_csv("test_data.csv", index=False)

# 2. Initialize the required 5 models
models = {
    "Logistic_Regression": LogisticRegression(),
    "Decision_Topic_Tree": DecisionTreeClassifier(max_depth=5),
    "K-Nearest_Neighbor": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

os.makedirs("model", exist_ok=True)

# 3. Train, Evaluate, and Save
performance_metrics = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else preds
    
    # Calculate required 6 evaluation metrics
    metrics = {
        "ML Model Name": name.replace("_", " "),
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": round(roc_auc_score(y_test, probs), 4),
        "Precision": round(precision_score(y_test, preds, average='macro'), 4),
        "Recall": round(recall_score(y_test, preds, average='macro'), 4),
        "F1": round(f1_score(y_test, preds, average='macro'), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4)
    }
    performance_metrics.append(metrics)
    
    # Export trained models & pipeline artifacts using joblib
    with open(f"model/{name}.pkl", "wb") as f:
        joblib.dump({"model": model, "scaler": scaler}, f)

# Display tabular summary to easily copy into your README
summary_df = pd.DataFrame(performance_metrics)
print("\n=== Assignment Comparison Table ===")
print(summary_df.to_markdown(index=False))
