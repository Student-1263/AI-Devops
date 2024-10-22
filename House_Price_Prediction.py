# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv('house_data.csv')

# Check if columns are concatenated, then split them
if 'size,   bedrooms,    age,    price' in data.columns:
    data = data['size,   bedrooms,    age,    price'].str.split(',', expand=True)
    data.columns = ['size', 'bedrooms', 'age', 'price']

# Proceed with your existing code
X = data[['size', 'bedrooms', 'age']]  # Features
y = data['price']  # Target

# Step 2: Load the dataset
# Use the correct delimiter for a tab-separated file
data = pd.read_csv('house_data.csv', delimiter='\t')

# Step 3: Data Preprocessing
# Print the column names to ensure there are no errors
print("Available columns:", data.columns)

# Separate features (X) and target variable (y)
X = data[['size', 'bedrooms', 'age']]  # Features
y = data['price']  # Target variable

# Step 4: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create a Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions on the test data
y_pred = model.predict(X_test)

# Create a DataFrame to compare Actual vs Predicted prices
comparison = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred})
pd.set_option('display.float_format', '{:.2f}'.format)

# Display the first few rows of the comparison
print("\nHouse Price Predictions vs Actual Prices:")
print(comparison.head())


