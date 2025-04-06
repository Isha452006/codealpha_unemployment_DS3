from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Load the trained model and transformer
with open('polynomial_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('poly_transformer.pkl', 'rb') as f:
    poly = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    # Get the date from the POST request
    data = request.get_json(force=True)
    date_str = data['date']
    
    # Convert the date string to a datetime object
    date_obj = pd.to_datetime(date_str)
    
    # Convert the date to ordinal
    date_ordinal = np.array([[date_obj.toordinal()]])
    
    # Transform the date using the same polynomial transformation
    date_poly = poly.transform(date_ordinal)
    
    # Make the prediction
    prediction = model.predict(date_poly)
    
    # Return the prediction as a JSON response
    return jsonify({'predicted_unemployment_rate': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
