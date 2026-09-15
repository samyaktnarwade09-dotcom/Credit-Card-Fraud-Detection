#  Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using **Logistic Regression** and **Random Forest**, with special focus on handling highly imbalanced data using **SMOTE**.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![Imbalanced-learn](https://img.shields.io/badge/SMOTE-Imbalanced--learn-red)

---

##  Project Overview

Credit card fraud detection is a classic **imbalanced classification** problem.  
Fraudulent transactions make up less than **0.2%** of the total data, making it challenging to build an effective model.

**Dataset**: Credit Card Fraud Detection Dataset (Kaggle)  
- Total Transactions: **284,807**  
- Fraudulent Transactions: **492**  
- Features: `Time`, `Amount`, and 28 anonymized PCA features (`V1`–`V28`)

---

##  Approach

1. Exploratory Data Analysis
2. Feature Scaling using `StandardScaler`
3. Handling Class Imbalance with **SMOTE**
4. Model Training:
   - Logistic Regression
   - Random Forest Classifier
5. Model Evaluation (Classification Report, Confusion Matrix, ROC-AUC)
6. Threshold Tuning to improve Fraud Recall

---

##  Key Results (Random Forest - Threshold 0.3)

| Class   | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Normal  | 1.00      | 0.99   | 1.00     |
| Fraud   | 0.15      | **0.91** | 0.26   |

- **Fraud Recall**: 91% (successfully detected most fraudulent transactions)
- Confusion Matrix: `[[56371, 493], [9, 89]]`

---

##  How to Run

1. Clone this repository
2. Download `creditcard.csv` and place it in the project folder
3. Open the notebook in **Jupyter Notebook** or **Google Colab**
4. Run all cells

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
