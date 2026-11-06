# Breast Cancer Diagnosis: Classification Models & Streamlit Deployment

**Live Streamlit App:** [PASTE YOUR STREAMLIT COMMUNITY CLOUD LINK HERE]
**GitHub Repository:** https://github.com/likhipersonal/breast-cancer-ml-assg

---

## a. Problem Statement

Early and accurate diagnosis of breast cancer is critical. Given measurements computed from a digitised image of a fine-needle aspirate (FNA) of a breast mass, the task is to classify each case as **Malignant (0)** or **Benign (1)**.

This is a binary classification problem. Five classification models are trained on the same dataset and compared using six evaluation metrics. The trained models are saved and served through a Streamlit web app, which evaluates them on an uploaded test CSV **without retraining**.

## b. Dataset Description

| Item | Details |
|---|---|
| Dataset | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository (loaded through `sklearn.datasets.load_breast_cancer`) |
| Instances | 569 (357 Benign, 212 Malignant) |
| Features | 30 numeric features, no missing values |
| Target | `target`: 0 = Malignant, 1 = Benign |
| Train / Test split | 70% / 30% (`random_state=42`): 398 training rows, 171 test rows |

Ten real-valued measurements are computed for each cell nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. For each one, three values are stored: the **mean**, the **standard error** and the **worst** (largest) value. This gives 3 × 10 = 30 features. Example column names in `test_data.csv`: `mean radius`, `radius error`, `worst radius`.

**Preprocessing:** features are standardised with `StandardScaler`, fitted on the training set only. The fitted scaler is saved as `model/scaler.pkl` and reused by the app.

## c. Repository Structure

```
breast-cancer-ml-assg/
│-- app.py              # Streamlit application
│-- train.py            # Trains the 5 models, saves them, creates test_data.csv
│-- requirements.txt    # Dependencies
│-- README.md
│-- test_data.csv       # Held-out test set (171 rows, 30 features + target)
│-- model/
    │-- scaler.pkl
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- support_vector_machine.pkl
```

**Run locally**

```bash
pip install -r requirements.txt
python train.py          # optional: regenerates model/ and test_data.csv
streamlit run app.py
```

**Streamlit app features**
- Upload a test CSV (it must contain the 30 features and the `target` column)
- Dropdown to choose one of the 5 models
- Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model
- Confusion matrix and classification report
- A tab comparing all 5 models on the uploaded data

## d. Models Used

All models use the same train/test split and the same scaled features.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=10000` |
| Decision Tree | `max_depth=5` |
| K-Nearest Neighbors | `n_neighbors=5` |
| Naive Bayes | Gaussian |
| Support Vector Machine | RBF kernel, `probability=True` |

### Comparison Table (171-row test set)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9979 | 0.9907 | 0.9815 | 0.9860 | 0.9625 |
| Decision Tree | 0.9532 | 0.9367 | 0.9630 | 0.9630 | 0.9630 | 0.8995 |
| KNN | 0.9591 | 0.9787 | 0.9633 | 0.9722 | 0.9677 | 0.9118 |
| Naive Bayes | 0.9357 | 0.9927 | 0.9450 | 0.9537 | 0.9493 | 0.8614 |
| Support Vector Machine | 0.9766 | 0.9966 | 0.9815 | 0.9815 | 0.9815 | 0.9497 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best model on this dataset. Once the features are scaled, the two classes are almost linearly separable, so a linear boundary does very well. It has the highest accuracy, AUC, F1 and MCC, with only 3 mistakes out of 171 cases. |
| Decision Tree | Weakest on AUC (0.9367). A single depth-5 tree outputs coarse probabilities, which hurts ranking quality, and axis-aligned splits struggle with the many correlated size features. Precision and recall are balanced, but MCC is clearly below the other models. |
| KNN | Strong and consistent (accuracy 0.9591). It works well because similar tumours cluster together in scaled feature space. Its recall of 0.9722 means few malignant-vs-benign mix-ups. It is slightly behind LR and SVM, probably because the fixed k=5 is sensitive to the nearest few points. |
| Naive Bayes | Lowest accuracy (0.9357) and MCC (0.8614), but a high AUC (0.9927), so its probability ranking is good while its default 0.5 cut-off is less accurate. Radius, perimeter and area are strongly correlated, which breaks the independence assumption of Naive Bayes. |
| Support Vector Machine | Very close to Logistic Regression (accuracy 0.9766, AUC 0.9966, MCC 0.9497). The RBF kernel and margin maximisation give a smooth, robust boundary. It matches the top model on recall and is a good alternative. |

### Overall Winner

**Logistic Regression** is the best model for this dataset. It ranks first in Accuracy (0.9825), AUC (0.9979), Precision (0.9907), F1 (0.9860) and MCC (0.9625), and ties for first in Recall (0.9815) with the SVM. Because the classes are close to linearly separable after scaling, the simplest model generalises best here, and it is also the easiest to interpret.

---

*Note: this is an educational project and is not a medical diagnostic tool.*