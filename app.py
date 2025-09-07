from flask import Flask, request, render_template
import joblib

# Load model & vectorizer
model = joblib.load("svm_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        text = request.form["text"]
        X = vectorizer.transform([text])
        pred = model.predict(X)[0]
label = "Fake" if pred == 1 else "Real"
prediction = f"Prediction: {label}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

