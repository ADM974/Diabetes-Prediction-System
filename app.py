import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os

# Configure Streamlit
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🏥",
    layout="wide"
)

# Cache the model to speed up predictions
@st.cache_resource
def load_model():
    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load diabetes dataset
    df = pd.read_csv(os.path.join(current_dir, "diabetes.csv"))
    
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

# Load model when app starts
model, accuracy = load_model()

def main():
    # Title and description
    st.title("Diabetes Prediction System")
    st.write("""
    This application predicts the likelihood of diabetes based on medical measurements.
    Enter your medical information below to get a prediction.
    
    The model is trained on the Pima Indians Diabetes Database and uses logistic regression.
    """)

    # Display accuracy
    st.sidebar.markdown("### Model Performance")
    st.sidebar.info(f"Model Accuracy: {accuracy}%")

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
            glucose = st.number_input("Glucose Level", min_value=0, value=100)
            blood_pressure = st.number_input("Blood Pressure", min_value=0, value=70)
            skin_thickness = st.number_input("Skin Thickness", min_value=0, value=20)

        with col2:
            insulin = st.number_input("Insulin Level", min_value=0, value=100)
            bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
            pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
            age = st.number_input("Age", min_value=0, value=30)

        # Prediction button
        submitted = st.form_submit_button("Predict")

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

            # Make prediction
            prediction = model.predict([data])[0]
            probability = model.predict_proba([data])[0][1]
            
            # Display result
            if prediction == 1:
                st.error(f"Prediction: Positive (Diabetes Likely)")
                st.write(f"Probability: {probability:.2%}")
                st.write("""
                **Recommendations:**
                - Consult with a healthcare professional
                - Consider lifestyle changes
                - Monitor blood sugar levels
                """)
            else:
                st.success(f"Prediction: Negative (No Diabetes)")
                st.write(f"Probability: {1 - probability:.2%}")
                st.write("""
                **Recommendations:**
                - Maintain healthy lifestyle
                - Regular check-ups
                - Balanced diet and exercise
                """)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
