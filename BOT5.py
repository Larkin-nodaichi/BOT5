import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import os
import kagglehub

# Upload kaggle.json and set up Kaggle API
def setup_kaggle(kaggle_file):
    if kaggle_file:
        kaggle_dir = os.path.expanduser("~/.kaggle")
        os.makedirs(kaggle_dir, exist_ok=True)  # Create the Kaggle directory
        with open(os.path.join(kaggle_dir, "kaggle.json"), "wb") as f:
            f.write(kaggle_file.getvalue())
        os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)  # Set permissions
    else:
        raise FileNotFoundError("kaggle.json file not uploaded.")

# Download dataset
def download_dataset():
    path = kagglehub.dataset_download("ankushpanday1/global-road-accidents-dataset")
    st.write("Path to dataset files:", path)
    return path

# Load the dataset
def load_data():
    dataset_path = download_dataset() + '/global_road_accidents.csv'  # Adjust based on actual file name
    data = pd.read_csv(dataset_path)
    return data

def preprocess_data(data):
    data.fillna(method='ffill', inplace=True)
    data['target'] = data['target'].astype(int)  # Replace 'target' with the appropriate column
    features = data.drop(columns=['target'])  # Replace 'target' with the appropriate column
    target = data['target']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler, features.columns, target

def build_models(X_train, y_train):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = cross_val_score(model, X_train, y_train, cv=5)
        model_scores[name] = np.mean(scores)
    return models, model_scores

def display_results(model_scores, X_test, y_test, best_model):
    st.write("### Model Performance")
    st.write(pd.DataFrame(model_scores.items(), columns=['Model', 'Cross-Validation Accuracy']))

    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    st.write(cm)

    if len(np.unique(y_test)) == 2:  # Binary classification
        y_prob = best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        st.write("### ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f}')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc='lower right')
        st.pyplot(fig)

# Streamlit UI
def main():
    st.title("Supervised Learning Model and Prototype")

    st.sidebar.title("Options")
    option = st.sidebar.radio("Choose a Phase", ["Upload Kaggle API Key", "Download Dataset", "Load Data", "Preprocess Data", "Build Models", "Evaluate Model", "Prototype"])

    if option == "Upload Kaggle API Key":
        st.header("Kaggle API Key")
        st.write("Please upload your kaggle.json file.")
        kaggle_file = st.file_uploader("Upload kaggle.json", type=["json"])
        if st.button("Set Up Kaggle"):
            if kaggle_file is not None:
                setup_kaggle(kaggle_file)
                st.success("Kaggle API has been set up successfully.")
            else:
                st.error("Please upload a valid kaggle.json file.")

    elif option == "Download Dataset":
        st.header("Download Dataset")
        st.write("Downloading the dataset from Kaggle...")
        download_dataset()
        st.write("Dataset downloaded successfully!")

    elif option == "Load Data":
        st.header("Dataset")
        data = load_data()
        st.write(data.head())

    elif option == "Preprocess Data":
        st.header("Preprocessing")
        data = load_data()
        X_train, X_test, y_train, y_test, scaler, feature_cols, target = preprocess_data(data)
        st.write("Data has been preprocessed.")
        st.write(f"Training samples: {X_train.shape[0]} Testing samples: {X_test.shape[0]}")

    elif option == "Build Models":
        st.header("Model Building")
        data = load_data()
        X_train, X_test, y_train, y_test, _, _, _ = preprocess_data(data)
        models, model_scores = build_models(X_train, y_train)
        st.write("Models built successfully!")
        st.write(model_scores)
        best_model_name = max(model_scores, key=model_scores.get)
        best_model = models[best_model_name]
        joblib.dump(best_model, 'best_model.pkl')
        st.write(f"Best Model: {best_model_name} saved!")

    elif option == "Evaluate Model":
        st.header("Model Evaluation")
        data = load_data()
        X_train, X_test, y_train, y_test, scaler, _, _ = preprocess_data(data)
        best_model = joblib.load('best_model.pkl')
        _, model_scores = build_models(X_train, y_train)
        display_results(model_scores, X_test, y_test, best_model)

    elif option == "Prototype":
        st.header("Prototype")
        st.write("Enter input values to predict using the best model.")
        data = load_data()
        _, X_test, _, _, scaler, feature_cols, _ = preprocess_data(data)
        user_input = []

        for col in feature_cols:
            value = st.number_input(f"{col}", value=float(X_test[0][list(feature_cols).index(col)]))
            user_input.append(value)

        if st.button("Predict"):
            user_data = scaler.transform([user_input])
            best_model = joblib.load('best_model.pkl')
            prediction = best_model.predict(user_data)
            st.write(f"Prediction: {prediction[0]}")

if __name__ == "__main__":
    main()
