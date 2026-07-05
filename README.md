# Heart Disease Prediction - Machine Learning Capstone Project (Week 4)

This repository contains the final **Major Capstone Project** for Week 4, focusing on building an end-to-end Machine Learning classification pipeline. The project predicts whether a patient is likely to develop heart disease based on various clinical parameters.

---

## Project Overview
Cardiovascular diseases are a leading cause of global mortality. This project leverages a structured clinical dataset to train a predictive model capable of classifying health risks. It implements data ingestion, automated categorical variable encoding, numerical scaling, model evaluation, and probabilistic diagnostics.

---

## Machine Learning Pipeline Architecture

### 1. Data Processing & One-Hot Encoding
* **Automated Feature Ingestion**: Fetches clinical features dynamically via unified backup web mirrors.
* **Categorical Handling (`pd.get_dummies`)**: Seamlessly converts string categorical vectors (such as chest pain variants, thal states) into optimized numeric flags (0 and 1) to eliminate structural data constraints.
* **Feature Standardization**: Implements `StandardScaler` to handle continuous variables, avoiding structural mathematical skewness during training.

### 2. Model Training
* **Algorithm**: Optimized **Decision Tree Classifier** (`max_depth=4`) to maximize feature separation while maintaining generalization boundaries (preventing overfitting).
* **Data Split**: Strict 80-20 Train-Test split validation parameters enforced.

---

##  Model Evaluation & Diagnostic Performance

The model's testing criteria produced highly efficient classification stats across both parameters:

### Classification Matrix Breakdown:
* **Overall Accuracy**: **79%** across all test features.
* **Class 0 (No Disease)**: Precision: **88%** | Recall: **82%** | F1-Score: **85%**
* **Class 1 (Heart Disease Risk)**: Precision: **60%** | Recall: **71%** | F1-Score: **65%**

### Advanced Statistical Visualizations Included:
1. **Confusion Matrix**: Maps absolute performance values to visually inspect True Negatives (**36** cases) and True Positives (**12** cases).
2. **ROC-AUC Diagnostics**: Plots the True Positive Rate against the False Positive Rate. The model achieved a powerful **AUC score of 0.83**, showcasing strong dynamic sensitivity thresholds.

---

## Repository Structure
* **`heart_disease_capstone_project.ipynb`**: Complete Python script containing feature transformations, model training, and performance plots.
* **`heart_disease_dataset.csv`**: The clean numerical dataset generated from the feature engineering pipeline.

---

## Technical Ecosystem Used
* **Languages**: Python
* **Data Core**: Pandas, NumPy
* **Analysis & Modeling**: Scikit-Learn (`tree.DecisionTreeClassifier`, `preprocessing.StandardScaler`)
* **Visualization Engine**: Matplotlib
