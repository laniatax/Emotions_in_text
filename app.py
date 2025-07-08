from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    emotion = "позитив" if prediction == 1 else "негатив"
    return jsonify({"emotion": emotion})

if __name__ == "__main__":
    app.run(debug=True)