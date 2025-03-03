import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import requests
import io
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# URL of the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip"

# Download the ZIP file
response = requests.get(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.content))

# Extract the ZIP contents to a folder
extract_path = "CCPP_Dataset"
zip_file.extractall(extract_path)

# Find the actual Excel file inside the extracted folder
excel_filepath = None
for root, dirs, files in os.walk(extract_path):
    for file in files:
        if file.endswith(".xls") or file.endswith(".xlsx"):
            excel_filepath = os.path.join(root, file)
            break

# Ensure the Excel file was found
if excel_filepath is None:
    raise FileNotFoundError("No Excel file found in the extracted ZIP folder!")

print(f"Excel file found: {excel_filepath}")

# Read the extracted Excel file
df = pd.read_excel(excel_filepath, engine="openpyxl")

# Rename columns for better understanding
df.columns = ["Temperature", "Exhaust Vacuum", "Ambient Pressure", "Relative Humidity", "Power Output"]

# Display dataset information
print("Dataset Head:\n", df.head())
print("\nDataset Info:\n", df.info())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Data Visualization
sns.pairplot(df)
plt.show()

# Splitting features and target variable
X = df.drop(columns=["Power Output"])
y = df["Power Output"]

# Split dataset into training and testing sets (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Predict Power Output for a new input
new_data = np.array([[25.0, 54.3, 1013.0, 60.0]])  # Example input
predicted_output = model.predict(new_data)

print("\nPredicted Power Output for new input:", predicted_output[0])

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
plt.xlabel("Actual Power Output")
plt.ylabel("Predicted Power Output")
plt.title("Actual vs Predicted Power Output")
plt.grid()
plt.show()
