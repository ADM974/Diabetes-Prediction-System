import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import urllib.request
import logging
import tempfile

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
        .stProgress {
            margin-bottom: 20px;
        }
        .input-help {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

# Cache the model to speed up predictions
def load_model():
    try:
        logger.info("Starting model loading...")
        
        # Download dataset to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "diabetes.csv")
            
            # Download dataset
            logger.info("Downloading dataset...")
            dataset_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
            urllib.request.urlretrieve(dataset_url, dataset_path)
            
            # Load diabetes dataset with proper column names
            column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            
            # Load dataset with column names
            df = pd.read_csv(dataset_path, header=None, names=column_names)
            logger.info(f"Dataset shape: {df.shape}")
            
            # Clean data: replace 0 values in certain columns with mean
            zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
            for column in zero_not_accepted:
                # Replace 0 with NaN first
                df[column] = df[column].replace(0, np.nan)
                # Calculate mean
                mean = df[column].mean(skipna=True)
                # Fill NaN with mean
                df[column] = df[column].fillna(mean)
                logger.info(f"Cleaned {column} with mean: {mean}")
            
            # Prepare data
            X = df.drop('Outcome', axis=1)
            y = df['Outcome']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            logger.info("Training model...")
            model = LogisticRegression(max_iter=1000, solver='lbfgs')
            model.fit(X_train, y_train)
            
            # Calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
            
            logger.info(f"Model trained successfully with accuracy: {accuracy}%")
            return model, accuracy
    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}", exc_info=True)
        st.error("Failed to load model. Please try again later.")
        return None, 0

# Load model when app starts
model, accuracy = None, 0

# Check if model is loaded
if model is None:
    try:
        model, accuracy = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")

# Main app function
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
    
    **About the measurements:**
    - Skin Thickness: Measured using a caliper at the triceps area (usually by healthcare professionals)
    - Insulin: Measured through blood tests (requires medical supervision)
    - Diabetes Pedigree Function: A score based on family history of diabetes
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
                "Glucose Level (mg/dL)",
                min_value=0,
                value=100,
                help="Plasma glucose concentration 2 hours after oral glucose tolerance test"
            )
            
            systolic = st.number_input(
                "Systolic Blood Pressure (mm Hg)",
                min_value=0,
                value=120,
                help="Top number of your blood pressure reading"
            )
            
            diastolic = st.number_input(
                "Diastolic Blood Pressure (mm Hg)",
                min_value=0,
                value=80,
                help="Bottom number of your blood pressure reading"
            )
            
            # Combine systolic and diastolic into one input
            blood_pressure = systolic
            
        with col2:
            skin_thickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0,
                value=20,
                help="Triceps skin fold thickness (measured by healthcare professionals)"
            )
            
            insulin = st.number_input(
                "Insulin Level (mu U/ml)",
                min_value=0,
                value=100,
                help="2-Hour serum insulin level (requires blood test)"
            )
            
            bmi = st.number_input(
                "BMI (kg/m²)",
                min_value=0.0,
                value=25.0,
                step=0.1,
                help="Body mass index (weight in kg/(height in m)²)"
            )
            
            pedigree_function = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0,
                value=0.5,
                step=0.01,
                help="Score based on family history of diabetes"
            )
            
            age = st.number_input(
                "Age (years)",
                min_value=0,
                value=30,
                help="Your current age"
            )

        # Add validation
        if st.form_submit_button("Predict"):
            try:
                # Validate inputs
                if glucose < 40 or glucose > 400:
                    st.error("Glucose level should be between 40-400 mg/dL")
                    return
                
                if systolic < 70 or systolic > 200:
                    st.error("Systolic blood pressure should be between 70-200 mm Hg")
                    return
                
                if diastolic < 40 or diastolic > 120:
                    st.error("Diastolic blood pressure should be between 40-120 mm Hg")
                    return
                
                if skin_thickness < 0 or skin_thickness > 100:
                    st.error("Skin thickness should be between 0-100 mm")
                    return
                
                if insulin < 0 or insulin > 800:
                    st.error("Insulin level should be between 0-800 mu U/ml")
                    return
                
                if bmi < 10 or bmi > 60:
                    st.error("BMI should be between 10-60 kg/m²")
                    return
                
                if age < 0 or age > 120:
                    st.error("Age should be between 0-120 years")
                    return

                # Prepare data for prediction
                data = [
                    pregnancies,
                    glucose,
                    blood_pressure,  # Using systolic as the main BP value
                    skin_thickness,
                    insulin,
                    bmi,
                    pedigree_function,
                    age
                ]

                if model is None:
                    st.error("Model failed to load. Please try again later.")
                    return

                # Show loading progress
                with st.spinner("Making prediction..."):
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
