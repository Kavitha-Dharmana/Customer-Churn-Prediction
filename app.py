
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and feature columns
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        customer = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        customer_df = pd.DataFrame([customer])
        customer_encoded = pd.get_dummies(
            customer_df, drop_first=True, dtype=int
        )
        customer_encoded = customer_encoded.reindex(
            columns=model_columns, fill_value=0
        )

        prediction = model.predict(customer_encoded)[0]
        probability = model.predict_proba(customer_encoded)[0][1] * 100

        if prediction == 1:
            result = "Customer may leave"
        else:
            result = "Customer may stay"

        return render_template(
            "index.html",
            result=result,
            probability=round(probability, 2)
        )

    except Exception as e:
        return f"Prediction error: {e}", 400


if __name__ == "__main__":
    app.run(debug=True)
