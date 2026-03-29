from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model("model.keras")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)
        prediction = model.predict(img_array)
        result = "Fake ❌" if prediction[0][0] > 0.5 else "Real ✅"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"result": str(e)})

if __name__ == "__main__":
    app.run(debug=True)