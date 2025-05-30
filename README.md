# House Price Predictor

A simple web application to predict house prices based on features like square footage, bedrooms, bathrooms, floors, and halls. The backend is built with Flask and scikit-learn, and the frontend is a clean HTML form.

## Features

- Predicts house price based on user inputs
- Model trained with synthetic data for demonstration
- Price predictions are clamped between ₹1,00,000 and ₹30,00,000
- Easy to run locally

## Demo

![image](https://github.com/user-attachments/assets/2a123fc4-5c23-4c12-b107-29d992fdd339)

# Model Exploration: House Price Predictor

## Model Overview

The house price prediction model is a **Random Forest Regressor** trained on synthetic data. It estimates house prices based on key features:
- Square footage (`sqft`)
- Number of bedrooms
- Number of bathrooms
- Number of floors
- Number of halls

The output is the **estimated price** of a house in Indian Rupees (₹), clamped between ₹1,00,000 and ₹30,00,000 for realistic results.

## Data Generation

Since real-world data was not available, we generated **synthetic data**. The price is calculated as a function of the features, with added random noise for realism:

```python
price = (
    sqft * 400 +
    bedrooms * 50000 +
    bathrooms * 40000 +
    floors * 25000 +
    halls * 20000 +
    np.random.randint(-50000, 50000, num)  # random noise
)
price = np.clip(price, 100000, 3000000)
```

This ensures that house prices increase with size and amenities, mimicking real-world trends.

## Model Training

- **Algorithm:** Random Forest Regressor (`sklearn.ensemble.RandomForestRegressor`)
- **Training/Test Split:** 80% train, 20% test
- **Features Used:** `sqft`, `bedrooms`, `bathrooms`, `floors`, `halls`
- **Target:** `price`
- **Evaluation Metric:** R² score (printed during training)

## Re-training the Model

To retrain or experiment:
1. Edit `train_model.py` to change feature weights, noise, or the number of samples.
2. Run:
   ```sh
   python train_model.py
   ```
   This will update `model.pkl` with a newly trained model.

## Exploring the Model

- **Feature Importance:**  
  You can inspect which features are most important to the model:
  ```python
  import pickle
  with open('model.pkl', 'rb') as f:
      model = pickle.load(f)
  print(model.feature_importances_)
  ```

- **Testing Predictions in Python:**
  ```python
  test_features = [[2000, 3, 2, 1, 1]]  # Example: 2000 sqft, 3 bedrooms, 2 bathrooms, 1 floor, 1 hall
  print(model.predict(test_features))
  ```

- **Experimenting:**  
  Adjust the weights in the price formula or the model parameters in `train_model.py` to see how predictions change.

## Limitations

- The model is trained on synthetic data, so its predictions are for demonstration only.
- For real applications, train on real, representative house price data.

## References

- [Random Forest Regression — scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
- [Pandas Documentation](https://pandas.pydata.org/)
- [NumPy Documentation](https://numpy.org/)

---

*Feel free to fork this project and experiment with the model!*


## How It Works

- **Frontend**: An HTML form collects house details from the user.
- **Backend**: A Flask app uses a trained RandomForestRegressor model (`model.pkl`) to predict the price.
- **Model**: Model is trained on randomly generated, but feature-dependent, synthetic data.

## Getting Started

### 1. Clone the repository

```sh
git clone https://github.com/muniasamyk/house-price-predictor.git
cd house-price-predictor
```

### 2. Install Requirements

Make sure you have Python 3.x and pip installed.

```sh
pip install flask scikit-learn pandas numpy flask-cors
```

### 3. Train Model (Optional)

If you want to retrain the model, run:

```sh
python train_model.py
```

This will generate a new `model.pkl` file.

### 4. Run the Flask Server

```sh
python app.py
```

The server will start at `http://localhost:5000`.

### 5. Open the Frontend

Open `index.html` in your web browser.

> **Tip:** For full functionality (due to CORS), use a local server for the frontend, or access via `http://localhost`.

---

## File Structure

```
house-price-predictor/
├── app.py                       # Flask backend
├── train_model.py               # Model/data generation and training script
├── model.pkl                    # Saved trained model
├── random_dataset_house_price.csv # Generated synthetic data
├── index.html                   # Frontend UI
├── .gitignore
└── README.md
```

## Customization

- To use your own dataset, adjust `train_model.py` to load your data.
- Modify weights/logic in `train_model.py` for more realistic predictions.

## License

MIT License

---

### Credits

Created by [muniasamyk](https://github.com/muniasamyk)
