from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the pre-trained model
    model = pickle.load(open('diabetes.pkl', 'rb'))
    
    # Get the input values from the form
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])
    
    # Make prediction
    prediction_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)
    prediction = model.predict(prediction_data)
    
    # Convert prediction to string
    if prediction:
        result = 'Positive'
    else:
        result = 'Negative'
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
