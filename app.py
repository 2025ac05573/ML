import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="ML Model Evaluator Workspace", layout="wide")
st.title("🎯 Machine Learning Classification & Evaluation Workbench")
st.write("Upload your assignment evaluation test data to swap models dynamically and observe live performance metrics.")

# Sidebar Configuration Layout
st.sidebar.header("📁 Step 1: Input Dataset")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

st.sidebar.header("🤖 Step 2: Select Classifier")
model_option = st.sidebar.selectbox(
    "Choose ML Model Strategy",
    ["Logistic Regression", "Decision Tree Classifier", "K-Nearest Neighbor Classifier", "Naive Bayes Classifier", "Random Forest"]
)

# Convert selector strings to match filesystem picklings
model_mapping = {
    "Logistic Regression": "Logistic_Regression",
    "Decision Tree Classifier": "Decision_Topic_Tree",
    "K-Nearest Neighbor Classifier": "K-Nearest_Neighbor",
    "Naive Bayes Classifier": "Naive_Bayes",
    "Random Forest": "Random_Forest"
}

if uploaded_file is not None:
    # Read test partition data
    test_data = pd.read_csv(uploaded_file)
    
    if 'target' not in test_data.columns:
        st.error("Error: The uploaded dataframe missing the column labeled 'target'.")
    else:
        X_test = test_data.drop(columns=['target'])
        y_test = test_data['target']
        
        # Load picked pipeline context
        model_filename = f"model/{model_mapping[model_option]}.pkl"
        try:
            with open(model_filename, "rb") as f:
                saved_artifact = pickle.load(f)
            
            model = saved_artifact["model"]
            scaler = saved_artifact["scaler"]
            
            # Apply transformation
            X_test_scaled = scaler.transform(X_test)
            
            # Form inference targets
            predictions = model.predict(X_test_scaled)
            probabilities = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else predictions
            
            # Component Columns Display
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📊 Primary Execution Metrics")
                metrics = {
                    "Metric Framework": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],
                    "Value Calculated": [
                        accuracy_score(y_test, predictions),
                        roc_auc_score(y_test, probabilities),
                        precision_score(y_test, predictions, average='macro'),
                        recall_score(y_test, predictions, average='macro'),
                        f1_score(y_test, predictions, average='macro'),
                        matthews_corrcoef(y_test, predictions)
                    ]
                }
                metrics_df = pd.DataFrame(metrics)
                st.dataframe(metrics_df.style.format({"Value Calculated": "{:.4f}"}), use_container_width=True)
                
            with col2:
                st.subheader("🧩 Confusion Matrix Visualization")
                cm = confusion_matrix(y_test, predictions)
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
                plt.ylabel('Actual Label Class')
                plt.xlabel('Predicted Label Class')
                st.pyplot(fig)
                
            st.subheader("📋 Detailed Classification Report Summary")
            report_dict = classification_report(y_test, predictions, output_dict=True)
            st.dataframe(pd.DataFrame(report_dict).transpose(), use_container_width=True)
            
        except FileNotFoundError:
            st.error(f"Missing pre-trained model binary file: `{model_filename}`. Run backend local script pipeline first.")
else:
    st.info("💡 Awaiting file submission. Upload your experimental verification `test_data.csv` using the left sidebar component.")
