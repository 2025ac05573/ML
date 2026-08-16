import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, matthews_corrcoef, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# 1. Load the Kaggle dataset (Use 'train.csv' from Kaggle)
# Make sure 'train.csv' downloaded from Kaggle is in your working directory
df = pd.read_csv("train.csv") 

# Rename Kaggle's 'price_range' column to 'target' to match your app structure
df = df.rename(columns={"price_range": "target"})

# Extract features and target column
feature_names = [col for col in df.columns if col != 'target']
X = df[feature_names]
y = df['target']

# Split dataset into training and evaluation sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the custom test dataset for your Streamlit upload requirement
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test.values
test_df.to_csv("test_data.csv", index=False)
print("✅ Created your assignment 'test_data.csv' successfully!")

# 2. Initialize the required 5 models
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
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
    
    # Handle multi-class probabilities safely for AUC score calculation
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test_scaled)
        # Using 'ovr' (One-vs-Rest) strategy because there are 4 target price classes
        auc = round(roc_auc_score(y_test, probs, multi_class='ovr'), 4)
    else:
        auc = 0.0000 
   
    # Calculate required 6 evaluation metrics using macro averaging for multi-class
    metrics = {
        "ML Model Name": name.replace("_", " "),
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "AUC": auc,
        "Precision": round(precision_score(y_test, preds, average='macro'), 4),
        "Recall": round(recall_score(y_test, preds, average='macro'), 4),
        "F1": round(f1_score(y_test, preds, average='macro'), 4),
        "MCC": round(matthews_corrcoef(y_test, preds), 4)
    }
    performance_metrics.append(metrics)
   
    # Export trained models & pipeline artifacts
    with open(f"model/{name}.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

# Display tabular summary to easily copy into your README file
summary_df = pd.DataFrame(performance_metrics)
print("\n=== Assignment Comparison Table ===")
print(summary_df.to_markdown(index=False))
