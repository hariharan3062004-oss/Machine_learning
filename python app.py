from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

print("Flask app file 'app.py' created and model/scaler loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    pass # Placeholder for prediction logic

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert input data to a pandas DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data])
        
        # Scale the input features
        # Use the global scaler object, which was loaded earlier
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)

        # Return prediction as JSON response
        # Convert numpy int64 to Python int for JSON serialization
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
