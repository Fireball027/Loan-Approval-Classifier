## Overview

The **Loan Application Classification Project** aims to classify loan applications as either **approved or rejected** using machine learning models. The dataset consists of **1,000 loan applicants** with **20 feature variables per applicant**. Out of these, **13 attributes** take discrete values, while **three attributes** accept continuous values. The goal is to extract the most influential attributes and employ them in an efficient classification model.

---

## Key Features

- **Data Preprocessing**: Cleans and balances the dataset for better model accuracy.
- **Feature Selection**: Identifies crucial attributes impacting loan classification.
- **Machine Learning Model**: Utilizes classification algorithms for predicting loan approval.
- **Model Evaluation**: Assesses accuracy, precision, recall, and confusion matrix.
- **Web Application**: Integrates with **Streamlit** for an interactive user experience.

---

## Project Files

### 1. `prediction.py`
This script processes the dataset, trains a classification model, and predicts loan approvals.

#### Key Components:

- **Data Preprocessing**:
  - Reads the German Credit Dataset.
  - Handles missing values and encodes categorical features.
  - Performs feature selection for improved performance.

- **Machine Learning Model**:
  - Splits the dataset into training and testing sets.
  - Implements classifiers such as **Logistic Regression, Random Forest, and SVM**.
  - Evaluates model performance using **accuracy and confusion matrix**.

- **Web Application**:
  - Deploys an interactive dashboard using **Streamlit**.
  - Allows users to input applicant details and receive loan approval predictions.

#### Example Code:
```python
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('german_credit_data.csv')

# Preprocessing
data.fillna(method='ffill', inplace=True)
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'loan_classifier.pkl')
```

### 2. `requirements.txt`
This file contains the dependencies required to run the project. Install them using:
```bash
pip install -r requirements.txt
```

Dependencies include:
- **numpy** (Numerical computations)
- **pandas** (Data handling)
- **scikit-learn** (Machine learning algorithms)
- **imbalanced-learn** (Handling imbalanced datasets)
- **streamlit** (Building an interactive web application)
- **joblib** (Model serialization)

---

## How to Run the Project

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model
```bash
python prediction.py
```

### Step 3: Run the Streamlit App
```bash
streamlit run prediction.py
```

### Step 4: Interact with the Web App
- Input applicant details.
- Get instant loan approval predictions.
- Visualize key attributes affecting loan classification.

---

## Future Enhancements

- **Deep Learning Model**: Implement neural networks for improved predictions.
- **Real-time Data Processing**: Connect with live credit databases for instant predictions.
- **Explainable AI (XAI)**: Provide insights on why an application was approved or rejected.
- **Cloud Deployment**: Deploy the Streamlit app on cloud platforms like AWS or Heroku.

---

## Conclusion

The **Loan Application Classification Project** provides an efficient way to automate loan approvals using **machine learning and data analytics**. By leveraging **Streamlit for interactivity**, this project bridges the gap between **data-driven decision-making** and **financial institutions**.

---

**Happy Predicting! 🚀**

