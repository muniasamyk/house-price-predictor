from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        float(data['sqft']),
        int(data['bedrooms']),
        int(data['bathrooms']),
        int(data['floors']),
        int(data['halls'])
    ]
    price = model.predict([features])[0]
    # Clamp predicted price between 1,00,000 and 30,00,000
    price = max(100000, min(5000000, price))
    return jsonify({'price': round(price, 2)})

if __name__ == '__main__':
    app.run(debug=True)