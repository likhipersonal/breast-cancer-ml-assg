import os
import pickle
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

print("Fetching Breast Cancer Wisconsin dataset from repository...")
# Load the real dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Combine to create a full dataframe view
df = X.copy()
df['target'] = y

# Split into Train and Test sets (70% train, 30% test)
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

# Save test dataset to a CSV file for Streamlit uploading
test_df.to_csv("test_data.csv", index=False)
print("Saved real test dataset to 'test_data.csv'")

# Separate features and target
X_train = train_df.drop(columns=['target'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target'])
y_test = test_df['target']

# Preprocessing: Scale features for stable distance/gradient steps
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Setup model folder
os.makedirs("model", exist_ok=True)

# Save the scaler object for app preprocessing consistency
with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Instantiate the 5 required models
models = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(probability=True, random_state=42)
}

performance_metrics = []

print("\nTraining and evaluating models...")
for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict labels and probability maps
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate all 6 mandatory metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    performance_metrics.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "AUC": round(auc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4)
    })
    
    # Save trained binary files
    filename = f"model/{name.lower().replace(' ', '_')}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"-> Successfully saved {name} to {filename}")

# Generate the performance comparison table
metrics_df = pd.DataFrame(performance_metrics)
print("\n=== Model Comparison Metrics Table ===")
print(metrics_df.to_markdown(index=False))