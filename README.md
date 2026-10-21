# Loan Default Prediction Using Gradient Boosting & XGBoost 💳📈

An advanced machine learning project that predicts whether a borrower will default on a loan, leveraging ensemble techniques (Gradient Boosting and XGBoost) on the Lending Club dataset.

## 📌 Project Overview
In the financial and banking sector, credit risk assessment is vital. Accurately predicting loan defaults helps institutions minimize financial losses and make informed lending decisions. This project builds a robust binary classification model using the Lending Club dataset (`accepted` loans) to distinguish between safe borrowers (`Fully Paid`) and high-risk defaulters (`Charged Off`).

## 🚀 Key Features & Workflow
1. **Data Filtering:** Cleaned and filtered over 1.4 million rows to focus strictly on completed loan outcomes (`Fully Paid` vs `Charged Off`), converting them into a binary target (`0` for Fully Paid, `1` for Charged Off).
2. **Missing Value Imputation:** Handled missing data intelligently by filling numerical columns with medians and categorical columns with mode/unknown tags.
3. **Categorical Encoding:** Utilized `LabelEncoder` to transform string-based categorical variables into machine-readable numeric formats.
4. **Ensemble Modeling:** Trained and evaluated state-of-the-art ensemble classifiers:
   - **Gradient Boosting Classifier**
   - **XGBoost Classifier** (Industry standard for tabular structured data)
5. **Model Evaluation:** Assessed model performance using Accuracy, Precision, Recall, F1-Score, and Classification Reports.

## 📊 Model Performance Results
Both ensemble models achieved exceptional performance on the test dataset:
* **Gradient Boosting:** Achieved **~99.2% Accuracy** with balanced high precision and recall across both classes.
* **XGBoost:** Achieved **~99.3% Accuracy**, demonstrating powerful pattern recognition and predictive stability for credit risk management.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** 
  * `Pandas` & `NumPy` (Data manipulation & cleaning)
  * `Scikit-Learn` (Model training, Label Encoding, Train-Test split, Evaluation metrics)
  * `XGBoost` (Optimized Gradient Boosting framework)
* **Environment:** Google Colab
  
## 💻 How to Run the Code
1. Clone the repository or open a new notebook in Google Colab.
2. Download the Lending Club `accepted_2007_to_2018Q4.csv` dataset.
3. Run the scripts step-by-step from data preprocessing to model evaluation to reproduce the results.

*Created as part of my Advanced Data Science Portfolio.*
