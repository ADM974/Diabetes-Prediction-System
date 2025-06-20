import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import urllib.request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
        .stTextInput, .stNumberInput {
            width: 100% !important;
        }
        .stFormSubmitButton {
            width: 100% !important;
        }
        .stMarkdown {
            text-align: justify;
        }
    </style>
    """, unsafe_allow_html=True)

# Cache the model to speed up predictions
@st.cache_resource
def load_model():
    try:
        # Download dataset if not exists
        dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        if not os.path.exists("diabetes.csv"):
            urllib.request.urlretrieve(dataset_url, "diabetes.csv")
        
        # Load diabetes dataset with proper column names
        column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                       'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
        # Load dataset with column names
        df = pd.read_csv("diabetes.csv", header=None, names=column_names)
        
        # Clean data: replace 0 values in certain columns with NaN
        zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
        for column in zero_not_accepted:
            df[column] = df[column].replace(0, np.nan)
            mean = int(df[column].mean(skipna=True))
            df[column] = df[column].replace(np.nan, mean)
        
        # Prepare data
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = LogisticRegression(max_iter=1000)
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

def main():
    # Title and description
    st.title("Diabetes Prediction System 🏥")
    st.markdown("""
    <div style='text-align: justify;'>
    This application predicts the likelihood of diabetes based on medical measurements. 
    It uses a logistic regression model trained on the Pima Indians Diabetes Database.
    
    **How to use:**
    1. Enter your medical information in the form below
    2. Click the "Predict" button
    3. View the prediction result and recommendations
    </div>
    """, unsafe_allow_html=True)

    # Display accuracy
    st.sidebar.markdown("### Model Performance")
    if model is not None:
        st.sidebar.info(f"Model Accuracy: {accuracy}%")
    else:
        st.sidebar.error("Model failed to load. Please try again later.")

    # Input form
    with st.form("prediction_form", clear_on_submit=False):
        st.markdown("### Enter Your Medical Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of times pregnant"
            )
            glucose = st.number_input(
                "Glucose Level",
                min_value=0,
                value=100,
                help="Plasma glucose concentration a 2 hours in an oral glucose tolerance test"
            )
            blood_pressure = st.number_input(
                "Blood Pressure",
                min_value=0,
                value=70,
                help="Diastolic blood pressure (mm Hg)"
            )
            skin_thickness = st.number_input(
                "Skin Thickness",
                min_value=0,
                value=20,
                help="Triceps skin fold thickness (mm)"
            )

        with col2:
            insulin = st.number_input(
                "Insulin Level",
                min_value=0,
                value=100,
                help="2-Hour serum insulin (mu U/ml)"
            )
            bmi = st.number_input(
                "BMI",
                min_value=0.0,
                value=25.0,
                step=0.1,
                help="Body mass index (weight in kg/(height in m)^2)"
            )
            pedigree_function = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                value=0.5,
                step=0.01,
                help="Diabetes pedigree function"
            )
            age = st.number_input(
                "Age",
                min_value=0,
                value=30,
                help="Age (years)"
            )

        # Prediction button
        submitted = st.form_submit_button("Predict 🔍")

    if submitted:
        try:
            # Prepare data for prediction
            data = [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                pedigree_function,
                age
            ]

            if model is None:
                st.error("Model failed to load. Please try again later.")
                return

            # Make prediction
            prediction = model.predict([data])[0]
            probability = model.predict_proba([data])[0][1]
            
            # Display result
            if prediction == 1:
                st.error("Prediction: Diabetes Likely 🚨")
                st.write(f"Probability: {probability:.2%}")
                st.markdown("""
                **Recommendations: 📝**
                - Consult with a healthcare professional as soon as possible
                - Consider lifestyle changes including diet and exercise
                - Monitor blood sugar levels regularly
                - Follow up with regular medical check-ups
                """)
            else:
                st.success("Prediction: No Diabetes 👍")
                st.write(f"Probability: {1 - probability:.2%}")
                st.markdown("""
                **Recommendations: 📝**
                - Maintain a healthy lifestyle
                - Regular medical check-ups
                - Balanced diet and regular exercise
                - Continue monitoring health indicators
                """)

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            st.error("Failed to make prediction. Please try again later.")

if __name__ == "__main__":
    main()
