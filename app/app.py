from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

# Load saved model pipeline
MODEL_PATH = "../models/pipe_xgb.joblib"
pipe = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Features order (same as your dataset except 'Class')
FEATURES = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9',
            'V10','V11','V12','V13','V14','V15','V16','V17','V18',
            'V19','V20','V21','V22','V23','V24','V25','V26','V27',
            'V28','Amount']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read CSV-like input string
        raw_input = request.form.get("features")
        values = [float(x.strip()) for x in raw_input.split(",")]

        if len(values) != len(FEATURES):
            return render_template("index.html", error=f"Expected {len(FEATURES)} values, got {len(values)}")

        row = pd.DataFrame([values], columns=FEATURES)

        # Get probability
        prob = pipe.predict_proba(row)[:,1][0]
        pred = int(prob >= 0.5)

        result = {
            "fraud_probability": round(prob, 4),
            "prediction": "Fraud" if pred == 1 else "Non-Fraud"
        }

        return render_template("index.html", prediction=result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True,)
