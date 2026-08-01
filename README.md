# LeafGuard AI - Plant Disease Detection System

Welcome to the LeafGuard AI project repository. This system is a deep-learning-based plant disease detection platform designed to assist farmers, growers, and agronomists. It features image classification, Grad-CAM model explainability overlays, and instant treatment recommendations.

This repository hosts implementation milestones for the project.

---

## Directory Structure

```text
leafguard_ai/
│
├── data/
│   └── plantvillage_metadata.csv   # Structured metadata of PlantVillage images with quality metrics
│
├── class_distribution.png          # Visual plot of crop and disease categories distribution
├── image_augmentation.png          # Visual demonstration of raw image rotation & flipping
├── sample_leaf.jpg                 # Simulated leaf image used for testing augmentations
├── confusion_matrices.png          # Confusion Matrix comparisons for ML classifiers
├── model_comparison.png            # Bar chart comparing Accuracy, Precision, Recall, F1
│
├── preprocessing.ipynb             # Phase 1: Tabular and CV image preprocessing pipeline
├── eda_feature_engineering.ipynb   # Phase 2: Feature engineering and visual EDA profiling
├── model_training_evaluation.ipynb # Phase 3: Model training, evaluation, and cross-validation
├── report.pdf                      # Phase 1 PDF Report (Milestone 1)
└── README.md                       # Documentation and project walkthrough
```

---

## Milestone 1: Dataset Selection & Preprocessing
*   **Preprocessing Pipeline:** Drop duplicates, impute missing values using median/mode, cap outliers using Interquartile Range (IQR) on exposure/contrast metrics, apply `StandardScaler` scaling, encode features (Label Encoding and One-Hot Encoding), and handle tabular class imbalance using `SMOTE-Tomek`.
*   **Jupyter Notebook:** [preprocessing.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/preprocessing.ipynb)
*   **Report Summary:** [report.pdf](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/report.pdf)

---

## Milestone 2: Feature Engineering & Exploratory Data Visualization
Focuses on creating rich attributes and visually profiling relationships within our leaf metadata prior to model development.
*   **Feature Engineering:** Created `resolution` (width * height), `aspect_ratio`, `density_score` (file size / resolution), and `brightness_blur_interaction`.
*   **Feature Selection:** Dropped `height` due to perfect multicollinearity (1.00 correlation with `width`). Calculated Mutual Information (MI) scores to rank feature importance.
*   **Jupyter Notebook:** [eda_feature_engineering.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/eda_feature_engineering.ipynb)

---

## Milestone 3: Model Training & Evaluation
Focuses on building classification models, validating their performance, and choosing the optimal classifier.

### 1. Model Training & Cross-Validation
*   **Train-Test Partition:** Implemented a stratified 80/20 train-test split to maintain class balance.
*   **Algorithms Evaluated:** Trained and compared **Logistic Regression**, **Random Forest Classifier**, and **Support Vector Machine (SVM)**.
*   **Robustness Checking:** Applied **5-Fold Cross-Validation** on the training set to prevent overfitting.
    *   *Logistic Regression:* stable baseline but unable to fit non-linear relations.
    *   *Support Vector Machine:* high accuracy, boundary constructed via scaling margins.
    *   *Random Forest:* highest mean accuracy and F1-Score, showing great variance resilience.

### 2. Model Evaluation Results
*   **Metrics Tracked:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
*   **Plots Generated:**
    *   `confusion_matrices.png`: Shows side-by-side True/False matrices for all models.
    *   `model_comparison.png`: Shows bar comparisons of F1-Score, Accuracy, Precision, and Recall on the test set.
*   **Best Model Selected:** **Random Forest Classifier** is selected as the primary classifier for LeafGuard AI due to its high F1-Score, robustness to multi-crop variations, and natural resistance to pixel quality outliers.
*   **Jupyter Notebook:** [model_training_evaluation.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/model_training_evaluation.ipynb)

---

## How to Setup and Run

### 1. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn notebook nbformat
```

### 2. Run Notebooks
Open the project directory in your terminal and launch Jupyter:
```bash
jupyter notebook
```
Browse and execute the cells of:
*   `preprocessing.ipynb` (Phase 1)
*   `eda_feature_engineering.ipynb` (Phase 2)
*   `model_training_evaluation.ipynb` (Phase 3)
