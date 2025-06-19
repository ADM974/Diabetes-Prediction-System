from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import json

app = Flask(__name__)

class DiabetesPredictor:
    def __init__(self):
        self.load_data()
        self.train_model()

    def load_data(self):
        self.df = pd.read_csv(os.path.join(os.path.dirname(__file__), "diabetes.csv"))

    def train_model(self):
        X = self.df.drop('Outcome', axis=1)
        y = self.df['Outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model = LogisticRegression()
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        self.accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

    def predict(self, data):
        prediction = self.model.predict([data])[0]
        return prediction

# Initialize the predictor
diabetes_predictor = DiabetesPredictor()

@app.route('/')
def home():
    return render_template('index.html', accuracy=diabetes_predictor.accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        prediction = diabetes_predictor.predict([
            float(data['pregnancies']),
            float(data['glucose']),
            float(data['blood_pressure']),
            float(data['skin_thickness']),
            float(data['insulin']),
            float(data['bmi']),
            float(data['pedigree_function']),
            float(data['age'])
        ])
        result = "Positive" if prediction == 1 else "Negative"
        return jsonify({'result': result, 'prediction': prediction})
    except ValueError:
        return jsonify({'error': 'Please enter valid numbers for all fields'}), 400

if __name__ == '__main__':
    app.run(debug=True)
