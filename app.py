from flask import Flask, request, jsonify
from keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scalers once at startup
model = load_model("gru_model.keras")
X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")


@app.route("/")
def home():
    return "GRU Forecasting API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    amount = data["AMOUNT"]
    shock = data["SHOCK"]
    impact = data["IMPACT"]

    features = np.array([[amount, shock, impact]])

    features_scaled = X_scaler.transform(features)

    # Create dummy sequence of length 480
    sequence = np.repeat(
        features_scaled.reshape(1, 1, 3),
        480,
        axis=1
    )

    prediction_scaled = model.predict(sequence)

    prediction = y_scaler.inverse_transform(
        prediction_scaled
    )

    return jsonify({
        "prediction": float(prediction[0][0])
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)