from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Connect to mlflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load registered model version
MODEL_URI = "models:/LogisticRegressionModelRegistered./1"

model = mlflow.sklearn.load_model(MODEL_URI)

# get the feature names
FEATURE_NAMES = list(model.feature_names_in_)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message" : "Customer CHurn Prediction API is running"    })
    
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status" : "healthy"
    })

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Read incoming data
        customer_data = request.get_json()
        if not isinstance(customer_data, dict):
             return jsonify({
                     "error" : "REquest body must be json"
             }), 400

        # Check for missing and unexpected col
        missing = [
            col for col in FEATURE_NAMES
            if col not in customer_data
        ]

        extra = [
            col for col in customer_data
            if col not in FEATURE_NAMES
        ]

        if missing or extra:
            return jsonify({
                     "error" : "Invalid input features",
                     "missing_features" : missing,
                     "extra_features" : extra
             }), 400


# convert JSON into DF
        input_df = pd.DataFrame(
            [customer_data],
            columns = FEATURE_NAMES
        )

        # Generate prediction
        prediction = model.predict(input_df)[0]

        # Generate probability
        prob = model.predict_proba(input_df)[0]

        classes = list(model.classes_)
        churn_index = classes.index(1)
        churn_prob = float(prob[churn_index])

        # return result
        return jsonify({
                       "prediction" : str(prediction),
                       "churn_prob" : round(churn_prob, 4) })

    except Exception as e:
        return jsonify({
            "eror" :  str(e)
        }), 500

if __name__ == "__main__":
    app.run(
         host ="127.0.0.1",
         port = 5001,
         debug = True
    )