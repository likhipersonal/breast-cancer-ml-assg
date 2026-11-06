import os
import re
import pickle

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
)

st.set_page_config(page_title="Breast Cancer Classifier Hub", page_icon="🔬", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}
CLASS_NAMES = ["Malignant (0)", "Benign (1)"]


# ---------------------------------------------------------------------------
# Column matching.
# This app is built for ONE dataset: Breast Cancer Wisconsin (Diagnostic).
# The same 30 features appear under different names depending on the source
# (sklearn: "mean radius", "radius error"; UCI/Kaggle: "radius_mean", "radius_se").
# Every name is reduced to (statistic, feature) so those variants all match.
# ---------------------------------------------------------------------------
STAT_TOKENS = {"mean": "mean", "se": "error", "error": "error",
               "stderr": "error", "worst": "worst"}


def canon(name):
    s = re.sub(r"[_\-\s]+", " ", str(name).strip().lower())
    stat, rest = None, []
    for tok in s.split():
        if tok == "standard":
            continue
        if tok in STAT_TOKENS and stat is None:
            stat = STAT_TOKENS[tok]
        else:
            rest.append(tok)
    return stat, " ".join(rest)


def align_features(df_features, expected_names):
    incoming = {}
    for col in df_features.columns:
        incoming.setdefault(canon(col), col)
    cols, missing = {}, []
    for exp in expected_names:
        src = incoming.get(canon(exp))
        if src is None:
            missing.append(exp)
        else:
            cols[exp] = pd.to_numeric(df_features[src], errors="coerce")
    if missing:
        return None, missing
    return pd.DataFrame(cols), []


def find_target(df):
    for c in df.columns:
        if c.strip().lower() in ("target", "diagnosis", "label", "class"):
            return c
    return None


def encode_target(y):
    """0 = Malignant, 1 = Benign (sklearn convention). Also accepts M/B."""
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    mapped = y.astype(str).str.strip().str.lower().map(
        {"m": 0, "malignant": 0, "b": 1, "benign": 1, "0": 0, "1": 1})
    if mapped.isna().any():
        raise ValueError("unrecognised target values")
    return mapped.astype(int)


def compute_metrics(y_true, y_pred, y_prob):
    auc = roc_auc_score(y_true, y_prob) if y_true.nunique() > 1 else float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


@st.cache_resource
def load_artifacts():
    """Load the saved scaler + all 5 trained models (no retraining anywhere)."""
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    models = {}
    for name, fname in MODEL_FILES.items():
        with open(os.path.join(MODEL_DIR, fname), "rb") as f:
            models[name] = pickle.load(f)
    return scaler, models


# ----------------------------------------------------------------------- UI
st.title("🔬 Breast Cancer Diagnostic Classifier Hub")
st.caption("Breast Cancer Wisconsin (Diagnostic) dataset · 30 features · Malignant (0) vs Benign (1) · "
           "pre-trained models, no retraining on uploaded data")

st.sidebar.header("📂 Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload test CSV (features + target)", type=["csv"])
selected_model_name = st.sidebar.selectbox("Select ML classifier", list(MODEL_FILES.keys()))

try:
    scaler, models = load_artifacts()
except FileNotFoundError:
    st.error("Saved models not found in the `model/` folder. Run `python train.py` first and commit the `model/` folder.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the saved models: {e}. "
             "This usually means the scikit-learn version differs from the one used in `train.py`; "
             "match the versions in requirements.txt and re-run `train.py`.")
    st.stop()

expected = list(getattr(scaler, "feature_names_in_", []))

with st.sidebar.expander("ℹ️ Expected CSV format"):
    st.write("The CSV must contain all 30 breast-cancer features plus a `target` column "
             "(0 = Malignant, 1 = Benign). Column names like `mean radius` or `radius_mean` "
             "are both accepted. Extra columns (e.g. `id`) are ignored.")
    if expected:
        st.code("\n".join(expected))

sample_path = os.path.join(BASE_DIR, "test_data.csv")
if os.path.exists(sample_path):
    with open(sample_path, "rb") as f:
        st.sidebar.download_button("⬇️ Download sample test_data.csv", f.read(),
                                   file_name="test_data.csv", mime="text/csv")

if uploaded_file is None:
    st.info("👈 Upload the breast-cancer `test_data.csv` from the sidebar to evaluate the models.")
    st.stop()

# ----------------------------------------------------------- validate input
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file as CSV: {e}")
    st.stop()
df.columns = df.columns.astype(str).str.strip()

target_col = find_target(df)
if target_col is None:
    st.error("The mandatory `target` column is missing from the uploaded file.")
    st.stop()

try:
    y_test = encode_target(df[target_col])
except Exception:
    st.error(f"The `{target_col}` column must hold 0/1 (0 = Malignant, 1 = Benign) or M/B labels.")
    st.stop()

X_test, missing = align_features(df.drop(columns=[target_col]), expected)
if missing:
    st.error("This app only works with the Breast Cancer Wisconsin (Diagnostic) dataset. "
             f"{len(missing)} required feature(s) were not found in your file: " + ", ".join(missing))
    st.caption("Columns found in your file: " + ", ".join(df.columns))
    st.stop()

if X_test.isna().any().any():
    st.error("Some feature values are empty or non-numeric. Please clean the CSV and upload again.")
    st.stop()

X_scaled = scaler.transform(X_test)
st.success(f"Loaded {len(df)} rows · {len(expected)} features matched · "
           f"class counts: Malignant={int((y_test == 0).sum())}, Benign={int((y_test == 1).sum())}")

# --------------------------------------------- evaluate all 5 saved models
results, preds, probs = {}, {}, {}
for name, mdl in models.items():
    preds[name] = mdl.predict(X_scaled)
    probs[name] = mdl.predict_proba(X_scaled)[:, 1]
    results[name] = compute_metrics(y_test, preds[name], probs[name])

tab_single, tab_compare = st.tabs([f"🎯 {selected_model_name}", "📈 Compare all 5 models"])

with tab_single:
    y_pred, m = preds[selected_model_name], results[selected_model_name]
    left, right = st.columns(2)
    with left:
        st.subheader("Evaluation metrics")
        for pair in (("Accuracy", "AUC"), ("Precision", "Recall"), ("F1", "MCC")):
            a, b = st.columns(2)
            a.metric(pair[0], f"{m[pair[0]]:.4f}")
            b.metric(pair[1], f"{m[pair[1]]:.4f}")
    with right:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="BuPu", xticklabels=CLASS_NAMES,
                    yticklabels=CLASS_NAMES, ax=ax)
        ax.set_ylabel("Actual label")
        ax.set_xlabel("Predicted label")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Classification report")
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=["Malignant", "Benign"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().style.format(precision=4), use_container_width=True)

with tab_compare:
    st.subheader("All models on the uploaded test data")
    comp = pd.DataFrame(results).T.rename_axis("ML Model Name")
    st.dataframe(comp.style.format(precision=4).highlight_max(axis=0, color="#3b5b3b"),
                 use_container_width=True)
    best = comp["MCC"].idxmax()
    st.caption(f"Highest MCC on this data: **{best}** ({comp.loc[best, 'MCC']:.4f})")
    st.bar_chart(comp[["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]])