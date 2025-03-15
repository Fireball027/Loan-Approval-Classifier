import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


def preprocess_data(df):
    if 'TARGET' not in df.columns:
        st.error("Error: No 'TARGET' column found in dataset.")
        return None, None, None

    X = df.drop(columns=['TARGET'])
    y = df['TARGET']

    # Encode categorical variables
    label_encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Normalize numerical data
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y, label_encoders


def balance_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return model, accuracy, report


def plot_graphs(df):
    st.subheader("Data Visualizations")

    # Histogram of numerical features
    st.write("Feature Distribution:")
    fig, ax = plt.subplots(figsize=(10, 5))
    df.hist(ax=ax, bins=30, figsize=(10, 5))
    st.pyplot(fig)

    # Correlation heatmap
    st.write("Feature Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)


def plot_class_distribution(y, title="Class Distribution"):
    st.write(title)
    fig, ax = plt.subplots()
    sns.countplot(x=y, ax=ax)
    st.pyplot(fig)


def main():
    st.title("ML Model Trainer")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.write("Data Preview:", df.head())

        plot_graphs(df)  # Plot initial graphs

        X, y, encoders = preprocess_data(df)
        if X is not None:
            st.write("Preprocessed Data:", X.head())

            plot_class_distribution(y, "Original Class Distribution")

            if st.checkbox("Apply SMOTE for balancing"):
                X, y = balance_data(X, y)
                st.write("Balanced Data Shape:", X.shape)
                plot_class_distribution(y, "Balanced Class Distribution")

            model, accuracy, report = train_model(X, y)

            st.write(f"Model Accuracy: {accuracy:.2f}")
            st.text("Classification Report:")
            st.text(report)


if __name__ == "__main__":
    main()
