python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model/model_updated.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    features = [data["home_form"], data["away_form"], data["home_elo"], data["away_elo"], data["bookmaker_home_win"]]
    prediction = model.predict([features])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
