import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pickle
import os

# Step 1: Load the dataset
df = pd.read_csv("C:/Users/HP/Downloads/Unemployment in India.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Filter data for Andhra Pradesh (assuming 'Region' is a column)
andhra_pradesh_data = df[df['Region'] == 'Andhra Pradesh']

# Convert 'Date' to ordinal (numeric representation)
andhra_pradesh_data['Date_ordinal'] = andhra_pradesh_data['Date'].apply(lambda x: x.toordinal())

# Step 2: Prepare the feature and target variables
# Feature: Date (in ordinal form)
X = andhra_pradesh_data[['Date_ordinal']]

# Target: Estimated Unemployment Rate
y = andhra_pradesh_data['Estimated Unemployment Rate (%)']

# Step 3: Create Polynomial Features
# Initialize the PolynomialFeatures transformer (degree=3)
poly = PolynomialFeatures(degree=3)

# Transform the feature (Date_ordinal) into polynomial features
X_poly = poly.fit_transform(X)

# Step 4: Train the Polynomial Regression Model
# Initialize the Linear Regression model
poly_model = LinearRegression()

# Fit the model to the polynomial features
poly_model.fit(X_poly, y)

# Step 5: Save the trained model and transformer
# Save the model to a .pkl file
with open('polynomial_regression_model.pkl', 'wb') as f:
    pickle.dump(poly_model, f)

# Save the PolynomialFeatures transformer to a .pkl file
with open('poly_transformer.pkl', 'wb') as f:
    pickle.dump(poly, f)

print("Model and transformer saved successfully.")

# Step 6: Load the saved model and transformer (for prediction later)
if os.path.exists('polynomial_regression_model.pkl') and os.path.exists('poly_transformer.pkl'):
    with open('polynomial_regression_model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)

    with open('poly_transformer.pkl', 'rb') as f:
        loaded_poly = pickle.load(f)

    print("Model and transformer loaded successfully.")

    # Step 7: Use the loaded model to make predictions
    # Example: Predict for a new date (2025-04-01)
    new_date = pd.to_datetime("2025-04-01")
    new_date_ordinal = new_date.toordinal()

    # Prepare the feature for the new date (convert to ordinal)
    new_data = np.array([[new_date_ordinal]])

    # Transform the new data using the same polynomial transformation
    new_data_poly = loaded_poly.transform(new_data)

    # Get the prediction from the loaded polynomial regression model
    predicted_unemployment = loaded_model.predict(new_data_poly)

    print(f"Predicted Unemployment Rate for {new_date.strftime('%Y-%m-%d')}: {predicted_unemployment[0]:.2f}%")
else:
    print("Model or transformer file not found.")
