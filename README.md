# House Price Predictor

A simple web application to predict house prices based on features like square footage, bedrooms, bathrooms, floors, and halls. The backend is built with Flask and scikit-learn, and the frontend is a clean HTML form.

## Features

- Predicts house price based on user inputs
- Model trained with synthetic data for demonstration
- Price predictions are clamped between ₹1,00,000 and ₹30,00,000
- Easy to run locally

## Demo

![image](https://github.com/user-attachments/assets/2a123fc4-5c23-4c12-b107-29d992fdd339)


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
