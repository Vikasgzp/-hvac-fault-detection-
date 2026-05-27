from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os

from tensorflow.keras.models import load_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("hvac_model.keras")

# Load scaler
scaler = joblib.load("scaler.pkl")

# Manual HVAC labels
labels = {
    0: "AHU Normal",
    1: "Coil Bias",
    2: "Coil Leakage",
    3: "Coil Stuck",
    4: "Damper Fault",
    5: "OA Bias",
    6: "Chiller Normal",
    7: "Chiller Bias",
    8: "Fouling",
    9: "Leakage",
    10: "Stuck"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check file
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            })

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No selected file"
            })

        # Save uploaded file
        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        # Read CSV
        df = pd.read_csv(filepath)

        # Remove label column if exists
        if "label" in df.columns:
            df_features = df.drop("label", axis=1)
        else:
            df_features = df

        # Scale features
        scaled_features = scaler.transform(df_features)

        # Predict
        predictions = model.predict(scaled_features)

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        # Convert IDs to labels
        predicted_labels = [
            labels[int(cls)]
            for cls in predicted_classes
        ]

        # Confidence scores
        confidence_scores = (
            np.max(predictions, axis=1) * 100
        )

        results = []

        for i in range(len(predicted_labels)):

            results.append({
                "row": int(i + 1),
                "prediction": str(predicted_labels[i]),
                "confidence": round(
                    float(confidence_scores[i]),
                    2
                )
            })

        return jsonify({
            "results": results
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(debug=True)