# Breast Cancer Diagnostic Analysis & Cloud Deployment Pipeline

## a. Problem Statement
The objective of this assignment is to design a high-accuracy, real-world machine learning classification and deployment architecture. Using cellular metric dimensions, the task is to classify biopsy cases as Malignant or Benign without re-training models during cloud validation cycles.

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI Machine Learning Repository / Kaggle).
* **Total Instances:** 569 instances
* **Total Feature Count:** 30 continuous clinical modeling features
* **Key Features Include:**
  1. `radius_mean`: Mean of distances from center to points on the perimeter.
  2. `texture_mean`: Standard deviation of gray-scale values.
  3. `perimeter_mean`: Mean size of the core tumor perimeter.
  4. `area_mean`: Mean area space of cell nucleus.
  5. `smoothness_mean`: Mean of local variation in radius lengths.
  6. `compactness_mean`: perimeter^2 / area - 1.0.
  7. `concavity_mean`: Severity of concave portions of the contour.
  8. `concave points_mean`: Number of concave portions of the contour.
  9. `symmetry_mean`: Geometric balance index.
  10. `fractal_dimension_mean`: "coastline approximation" - 1.
  *(Includes corresponding standard error 'se' and 'worst' maximized measurements across all 10 base parameters, totaling 30 distinct features).*
* **Target Label:** `target` (0: Malignant, 1: Benign).

## c. GitHub Repository Link
[Insert Your Live GitHub Repository Link URL Here]

## d. Models Used & Comparison Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9825 | 0.9977 | 0.9815 | 0.9907 | 0.9860 | 0.9626 |
| **Decision Tree** | 0.9415 | 0.9406 | 0.9533 | 0.9533 | 0.9533 | 0.8762 |
| **K-Nearest Neighbor** | 0.9591 | 0.9890 | 0.9554 | 0.9813 | 0.9682 | 0.9127 |
| **Naive Bayes** | 0.9357 | 0.9877 | 0.9364 | 0.9626 | 0.9493 | 0.8629 |
| **Support Vector Machine** | 0.9766 | 0.9974 | 0.9725 | 0.9907 | 0.9815 | 0.9503 |

### Performance Observations
* **Logistic Regression:** Achieves phenomenal performance on this scaled dataset. Linear boundaries cleanly split high-dimensional cancer features, yielding a near-perfect MCC score of 0.9626.
* **Decision Tree:** Separates features well but exhibits slightly lower test numbers due to typical rigid axis-aligned decision splitting, making it more prone to boundary variance.
* **KNN:** Captures neighborhood similarities effectively. It yields strong metrics across all categories, showing that similar tissue properties cluster tightly together.
* **Naive Bayes:** Shows solid baseline robustness but trails slightly because features like radius, perimeter, and area are highly correlated, which conflicts with Naive Bayes' assumption of independent features.
* **Support Vector Machine:** Excellent results due to robust margin maximizing features. It matches the top models closely with an outstanding 0.9974 AUC rating.

### Overall Winner for Dataset
**Logistic Regression** is the overall winner for this specific dataset. It achieves the highest **Accuracy (98.25%)**, **F1-Score (0.9860)**, and **MCC Score (0.9626)**, proving that a properly regularized linear surface is highly effective for high-dimensional diagnostic datasets.