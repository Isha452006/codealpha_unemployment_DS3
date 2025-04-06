import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the data (ensure the file path is correct)
df = pd.read_csv("C:/Users/HP/Downloads/Unemployment in India.csv")

# Clean column names by removing extra spaces
df.columns = df.columns.str.strip()

# Convert 'Date' column to datetime for better handling
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Filter data for Andhra Pradesh
andhra_pradesh_data = df[df['Region'] == 'Andhra Pradesh']

# Prepare data for linear regression (Date -> ordinal format)
andhra_pradesh_data['Date_ordinal'] = andhra_pradesh_data['Date'].apply(lambda x: x.toordinal())  # Convert date to ordinal

# Define the feature (X) and the target (y)
X = andhra_pradesh_data[['Date_ordinal']]  # Feature: Date (in ordinal form)
y = andhra_pradesh_data['Estimated Unemployment Rate (%)']  # Target: Unemployment rate

# Initialize the polynomial features (degree 3, you can change the degree if needed)
poly = PolynomialFeatures(degree=3)

# Apply polynomial features transformation to the Date feature (to capture non-linear relationship)
X_poly = poly.fit_transform(X)

# Train polynomial regression model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Predict using the polynomial regression model
andhra_pradesh_data['Predicted Unemployment Rate (Poly)'] = poly_model.predict(X_poly)

# Plot the original vs predicted unemployment rate with polynomial regression
plt.plot(andhra_pradesh_data['Date'], y, label='Original', marker='o')
plt.plot(andhra_pradesh_data['Date'], andhra_pradesh_data['Predicted Unemployment Rate (Poly)'], label='Predicted (Poly)', linestyle='--')
plt.title('Original vs Predicted Unemployment Rate (Polynomial Regression) in Andhra Pradesh')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Evaluate Polynomial Model
mae_poly = mean_absolute_error(y, andhra_pradesh_data['Predicted Unemployment Rate (Poly)'])
mse_poly = mean_squared_error(y, andhra_pradesh_data['Predicted Unemployment Rate (Poly)'])
r2_poly = r2_score(y, andhra_pradesh_data['Predicted Unemployment Rate (Poly)'])

# Print the evaluation metrics for the polynomial model
print(f"Polynomial Model - Mean Absolute Error (MAE): {mae_poly:.2f}")
print(f"Polynomial Model - Mean Squared Error (MSE): {mse_poly:.2f}")
print(f"Polynomial Model - R-squared (R²): {r2_poly:.2f}")
