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
│
├── preprocessing.ipynb             # Phase 1: Tabular and CV image preprocessing pipeline
├── eda_feature_engineering.ipynb   # Phase 2: Feature engineering and visual EDA profiling
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

### 1. Feature Engineering
We engineered the following features to capture spatial and file characteristics:
*   **`resolution`**: Image pixel resolution (`width` * `height`).
*   **`aspect_ratio`**: Proportional aspect ratio (`width` / `height`).
*   **`density_score`**: File size divided by resolution (`file_size_kb` / `resolution`). Serving as a proxy for color/frequency detail, as diseased leaves containing spots and lesions require higher compression detail.
*   **`brightness_blur_interaction`**: Product of brightness and blurriness (`brightness` * `blurriness_score`) to easily capture poor exposure/focus combinations.

### 2. Feature Selection & Importance
*   **Multicollinearity Treatment:** Dropped the feature `height` due to a perfect correlation coefficient of **1.00** with `width` (as all images are square).
*   **Mutual Information (MI):** Calculated Mutual Information scores (`mutual_info_classif`) for all features against the leaf health target (`status`). Plotted results as feature importance rankings, showing that engineered compression `density_score` and quality metrics provide high predictive values.

### 3. Visualizations Generated in Notebook
*   **Target Variable Distribution**: Bar plot showing healthy vs. diseased frequencies.
*   **Numerical Distributions**: KDE/Histograms showing normal distribution of brightness, right-skewed distribution of blurriness, and multimodal peaks of density scores.
*   **Crop Type Distribution**: Visualizing representation across crop species.
*   **Box Plots for Outlier Analysis**: Outliers in brightness and blurriness across crop types.
*   **Correlation Heatmap**: Inspecting correlation matrix coefficients.
*   **Scatter Plot**: Resolution vs. file size, revealing that diseased leaves cluster slightly higher in file size within resolution bands.
*   **Class-wise Box Plots**: Direct health status comparisons demonstrating that diseased leaves have higher Laplacian blurriness (spot edges) and slightly lower brightness (necrotic spots).
*   **Jupyter Notebook:** [eda_feature_engineering.ipynb](file:///C:/Users/Admin/.gemini/antigravity/scratch/leafguard_ai/eda_feature_engineering.ipynb)

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
