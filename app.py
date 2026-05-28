from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
import traceback

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

# Expected number of features
EXPECTED_FEATURES = 106


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # Check uploaded file
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded"
            }), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({
                "error": "No selected file"
            }), 400

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
            df = df.drop("label", axis=1)

        # Validate feature count
        if df.shape[1] != EXPECTED_FEATURES:

            return jsonify({
                "error": f"Expected {EXPECTED_FEATURES} features but got {df.shape[1]}"
            }), 400

        # Replace problematic values
        df = df.replace(
            ["#######", "######", "inf", "-inf"],
            np.nan
        )

        # Convert all columns to numeric
        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace infinite values
        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Replace NaN values
        df = df.fillna(0)

        # Debugging logs
        print("Shape:", df.shape)
        print("NaN count:", df.isna().sum().sum())
        print("Columns:", len(df.columns))

        # Force float conversion
        df = df.astype(float)

        # Scale features
        scaled_features = scaler.transform(df)

        # Predict
        predictions = model.predict(
            scaled_features,
            verbose=0
        )

        # Get predicted class IDs
        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        # Convert IDs to labels
        predicted_labels = [
            labels.get(
                int(cls),
                "Unknown"
            )
            for cls in predicted_classes
        ]

        # Confidence scores
        confidence_scores = (
            np.max(predictions, axis=1) * 100
        )

        # Build results
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
            "success": True,
            "total_rows": int(len(results)),
            "results": results
        })

    except Exception as e:

        print(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
