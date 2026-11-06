import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="UCI Breast Cancer Classifier Hub", layout="wide")

st.title("🔬 Breast Cancer Diagnostic Classifier Hub")
st.write("Upload the real validation dataset (`test_data.csv`) to test the 5 saved models instantly.")

# Sidebar Configuration Controls
st.sidebar.header("📂 Data & Model Upload")
uploaded_file = st.sidebar.file_uploader("Upload Test CSV File", type=["csv"])

model_mapping = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Support Vector Machine": "model/support_vector_machine.pkl"
}

selected_model_name = st.sidebar.selectbox("Select ML Classifier", list(model_mapping.keys()))

if uploaded_file is not None:
    # Read test input data
    df = pd.read_csv(uploaded_file)
    
    if 'target' not in df.columns:
        st.error("Error: The mandatory validation column 'target' is missing from the dataset.")
    else:
        # Separate data structures
        X_test = df.drop(columns=['target'])
        y_test = df['target']
        
        try:
            # Load the tracking scaler and saved model configurations
            with open("model/scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open(model_mapping[selected_model_name], "rb") as f:
                model = pickle.load(f)
                
            # Perform runtime scaling
            X_test_scaled = scaler.transform(X_test)
            
            # Form calculations
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 {selected_model_name} Performance Cards")
                
                # Metric computation
                metrics = {
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC Score": roc_auc_score(y_test, y_prob),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall": recall_score(y_test, y_pred),
                    "F1 Score": f1_score(y_test, y_pred),
                    "MCC Score": matthews_corrcoef(y_test, y_pred)
                }
                
                # Render Grid Metrics Cards
                c1, c2 = st.columns(2)
                c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                c2.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
                
                c3, c4 = st.columns(2)
                c3.metric("Precision", f"{metrics['Precision']:.4f}")
                c4.metric("Recall", f"{metrics['Recall']:.4f}")
                
                c5, c6 = st.columns(2)
                c5.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                c6.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
                
            with col2:
                st.subheader("🧩 Confusion Matrix Visualisation")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='BuPu', 
                            xticklabels=['Malignant (0)', 'Benign (1)'], 
                            yticklabels=['Malignant (0)', 'Benign (1)'], ax=ax)
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
                
            st.subheader("📋 Full Classification Report Matrix")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=4))
            
        except FileNotFoundError:
            st.error("Error: Pre-compiled weights are missing. Run `train.py` in your terminal environment first.")
else:
    st.info("💡 Complete the setup by picking and dropping your generated `test_data.csv` into the dashboard file uploader window located in the left sidebar layout.")