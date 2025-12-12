# Step 1: Import necessary libraries
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Step 2: Initialize Flask app
app = Flask(__name__)

# Step 3: Load the trained ML model
try:
    model = joblib.load('model/predictive_model.pkl')
except FileNotFoundError:
    print("❌ Model file not found! Please train the model and save it at 'model/predictive_model.pkl'")
    exit(1)

# Step 4: List of sensors used as input features for prediction
SENSORS = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15',
    'sensor_17', 'sensor_20', 'sensor_21'
]

# Step 5: Web UI - route to show the input form
@app.route('/')
def index():
    return render_template('index.html', sensors=SENSORS, prediction=None, error=None)

# Step 6: Improved error handling in UI prediction
@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    try:
        input_data = {}
        for sensor in SENSORS:
            value = request.form.get(sensor)
            if value is None or value.strip() == "":
                raise ValueError(f"Missing value for {sensor.replace('_', ' ')}")
            try:
                input_data[sensor] = float(value)
            except ValueError:
                raise ValueError(f"Invalid input for {sensor.replace('_', ' ')}. Please enter a valid number.")
        
        df = pd.DataFrame([input_data])
        prediction = int(model.predict(df)[0])

        return render_template('index.html', sensors=SENSORS, prediction=prediction, error=None)

    except Exception as e:
        print("❌ Prediction Error:", e)
        return render_template('index.html', sensors=SENSORS, prediction=None, error=str(e))

# Step 7: REST API - handle JSON POST requests
@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        missing_features = [f for f in SENSORS if f not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400

        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return jsonify({'failure_prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Step 8: Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
