import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import urllib.request

# Configure Streamlit
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

# Cache the model to speed up predictions
@st.cache_resource
def load_model():
    try:
        # Download dataset if not exists
        dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        if not os.path.exists("diabetes.csv"):
            urllib.request.urlretrieve(dataset_url, "diabetes.csv")
        
        # Load diabetes dataset
        df = pd.read_csv("diabetes.csv", header=None)
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                     'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        # Prepare data
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
        
        return model, accuracy
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, 0

# Load model when app starts
model, accuracy = load_model()

{{ ... }}
