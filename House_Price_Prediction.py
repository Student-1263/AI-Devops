# Step 1: Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 2: Load the dataset (assuming the file is comma-separated)
data = pd.read_csv('house_data.csv')

# Step 3: Data Preprocessing
# Print the column names to ensure they are correct
print("Available columns:", data.columns)

# Step 4: Separate features (X) and target variable (y)
X = data[['size', 'bedrooms', 'age']].astype(float)  # Ensure all values are numeric
y = data['price'].astype(float)  # Target variable as float

# Step 5: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create a Linear Regression model and train it
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 8: Create a DataFrame to compare Actual vs Predicted prices
comparison = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred})
pd.set_option('display.float_format', '{:.2f}'.format)  # Format float values

# Step 9: Display the first few rows of the comparison
print("\nHouse Price Predictions vs Actual Prices:")
print(comparison.head())





