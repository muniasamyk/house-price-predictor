import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

# Set random seed
np.random.seed(42)
num = 100

# Generate features
sqft      = np.random.randint(800, 4000, num)
bedrooms  = np.random.randint(1, 6, num)
bathrooms = np.random.randint(1, 3, num)
floors    = np.random.randint(1, 5, num)
halls     = np.random.randint(1, 3, num)

# Generate price as a function of the features + some noise
price = (
    sqft * 800 +
    bedrooms * 100000 +
    bathrooms * 80000 +
    floors * 75000 +
    halls * 50000 +
    np.random.randint(-50000, 50000, num)  # random noise
)

# Clamp price between 1,00,000 and 30,00,000 as per your requirements
price = np.clip(price, 100000, 5000000)

data = {
    "sqft": sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "floors": floors,
    "halls": halls,
    "price": price
}

df = pd.DataFrame(data)
df.to_csv("random_dataset_house_price.csv", index=False)

# Read the data
data1 = pd.read_csv("random_dataset_house_price.csv")

X = data1[['sqft', 'bedrooms', 'bathrooms', 'floors', 'halls']]
y = data1['price']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluate model (optional)
score = model.score(X_test, y_test)
print(f"Test R^2 score: {score:.2f}")

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)