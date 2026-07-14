import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Global variables for models and preprocessor
models = {}
preprocessor = None

def load_saved_models():
    global preprocessor, models
    print("Loading serialized models and preprocessor from disk...")
    
    # Load the preprocessor and models from the serialized joblib files
    preprocessor = joblib.load("Notebook/models/preprocessor.joblib")
    models['Linear Regression'] = joblib.load("Notebook/models/linear_regression_model.joblib")
    models['XGBoost'] = joblib.load("Notebook/models/xgboost_model.joblib")
    models['Random Forest'] = joblib.load("Notebook/models/random_forest_model.joblib")
    
    print("All models loaded successfully!")

# Load models on server startup
load_saved_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_name = data.get('model', 'Random Forest')
        
        # Extract features from request
        income = float(data.get('IncomeLevel', 3.5))
        age = float(data.get('PropertyAge', 25.0))
        rooms = float(data.get('TotalRooms', 5.0))
        bedrooms = float(data.get('TotalBedrooms', 1.0))
        pop = float(data.get('NeighborhoodPop', 1200.0))
        occupancy = float(data.get('AvgOccupancy', 3.0))
        lat = float(data.get('Latitude', 34.0))
        lon = float(data.get('Longitude', -118.0))
        # Calculate derived features
        rooms_per_household = float(data.get('RoomsPerHousehold', 2.0))
        bedrooms_ratio = float(data.get('BedroomsRatio', 0.2))
        
        # Bin age
        if age <= 15:
            age_bin = 'New'
        elif age <= 35:
            age_bin = 'Moderate'
        else:
            age_bin = 'Old'
            
        # Create input dataframe
        input_data = pd.DataFrame([{
            'IncomeLevel': income,
            'TotalRooms': rooms,
            'TotalBedrooms': bedrooms,
            'NeighborhoodPop': pop,
            'AvgOccupancy': occupancy,
            'Latitude': lat,
            'Longitude': lon,
            'RoomsPerHousehold': rooms_per_household,
            'BedroomsRatio': bedrooms_ratio,
            'PropertyAge_bins': age_bin
        }])
        
        # Transform using preprocessor
        input_processed = preprocessor.transform(input_data)
        
        # Predict price using loaded model
        model = models.get(model_name, models['XGBoost'])
        predicted_val = float(model.predict(input_processed)[0])
        
        # Cap at 0 if negative predictions happen
        predicted_price = max(0.0, predicted_val)
        
        return jsonify({
            'success': True,
            'predicted_price': round(predicted_price, 4),
            'formatted_price': f"${predicted_price * 100000:,.2f}" # Assuming target price unit is $100k
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    # Start flask server
    app.run(debug=True, port=5000)
