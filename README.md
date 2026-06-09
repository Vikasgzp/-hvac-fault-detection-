# HVAC Fault Detection using Deep Learning

## Overview

This project is an AI-powered HVAC (Heating, Ventilation, and Air Conditioning) Fault Detection System developed using TensorFlow and Deep Learning.

The model analyzes HVAC sensor data and predicts potential faults in HVAC systems. Users can upload a CSV file containing sensor readings, and the application automatically predicts fault categories along with confidence scores.

The application is deployed using Gradio on Hugging Face Spaces, providing an easy-to-use web interface for real-time fault prediction.

---

## Live Demo

🔗 **Hugging Face Space**
https://huggingface.co/spaces/Vikasgzp/hvac-fault-detection

---

## Features

* Upload HVAC sensor CSV files
* Automatic fault prediction
* Confidence score for each prediction
* User-friendly Gradio interface
* Deep Learning-based classification
* Supports multiple HVAC fault categories
* Real-time prediction results

---

## Supported Fault Classes

| Class ID | Fault Type     |
| -------- | -------------- |
| 0        | AHU Normal     |
| 1        | Coil Bias      |
| 2        | Coil Leakage   |
| 3        | Coil Stuck     |
| 4        | Damper Fault   |
| 5        | OA Bias        |
| 6        | Chiller Normal |
| 7        | Chiller Bias   |
| 8        | Fouling        |
| 9        | Leakage        |
| 10       | Stuck          |

---

## Project Structure

```text
hvac-fault-detection/
│
├── app.py
├── requirements.txt
├── hvac_model.keras
├── scaler.pkl
├── label_encoder.pkl
└── README.md
```

---

## Technologies Used

* Python
* TensorFlow / Keras
* Scikit-Learn
* Pandas
* NumPy
* Gradio
* Hugging Face Spaces

---

## Input Format

The uploaded CSV file must:

* Contain HVAC sensor features used during training
* Follow the same column structure as the training dataset
* Contain numerical feature values
* Have feature columns in the same order as the training data
* The `label` column is optional and will be ignored if present

---

## Output

The system returns:

* Predicted fault type
* Confidence score (%)
* Row-wise prediction results

### Example Output

| Row | Prediction | Confidence (%) |
| --- | ---------- | -------------- |
| 1   | AHU Normal | 98.45          |
| 2   | Leakage    | 92.18          |
| 3   | Fouling    | 95.67          |

---

## Model Information

* Framework: TensorFlow / Keras
* Task: Multi-Class Classification
* Number of Classes: 11
* Input: HVAC Sensor Data
* Output: HVAC Fault Category Prediction

---

## Deployment

The application is deployed on Hugging Face Spaces using Gradio.

Users can upload HVAC sensor CSV files directly through the web interface and receive fault predictions instantly.

---

## Future Improvements

* Interactive analytics dashboard
* Fault visualization charts
* Downloadable prediction reports
* Batch prediction export
* Real-time sensor monitoring
* Advanced model explainability
* REST API integration

---

## Project Team

This project was developed as a collaborative academic project by:

* Vikas Kushwaha
* Nitish Kumar Mahto
* Gugulothu Vijay
* Harsh
* Harshdeep Singh
* Harmanjot Kaur

---

## Institution

**Dr. B. R. Ambedkar National Institute of Technology (NIT) Jalandhar**

---

## Academic Program

**Minor Project under Bachelor of Technology (B.Tech)**

---

## Project Lead

**Vikas Kushwaha**
B.Tech, Computer Science & Engineering
Dr. B. R. Ambedkar National Institute of Technology (NIT) Jalandhar

### Connect

* GitHub: https://github.com/Vikasgzp
* LinkedIn: https://www.linkedin.com/in/vikasgzp

---

## License

This project is intended for academic, educational, and research purposes.
