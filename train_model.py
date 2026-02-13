import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
df = pd.read_csv('bodyfat.csv')

# Select features (based on the app.py - using density, abdomen, chest, weight, hip)
X = df[['Density', 'Abdomen', 'Chest', 'Weight', 'Hip']]
y = df['BodyFat']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model
with open('bodyfatmodel.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Model saved successfully!")