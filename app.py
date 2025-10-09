import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# ------------------ Config ------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "skin_detector.h5"
LABELS_PATH = BASE_DIR / "class_names.json"
RECS_PATH = BASE_DIR / "recommendations.json"
UPLOAD_FOLDER = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

UPLOAD_FOLDER.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["SECRET_KEY"] = "change-this-secret"

# ------------------ Globals ------------------
_model = None
_input_size = None
_class_names = None
_recommendations = {}

def load_model_and_labels():
    global _model, _input_size, _class_names, _recommendations

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
        shape = _model.input_shape[0] if isinstance(_model.input_shape, list) else _model.input_shape
        _, H, W, _ = shape
        _input_size = (H, W)

    if _class_names is None and LABELS_PATH.exists():
        with open(LABELS_PATH, "r") as f:
            _class_names = json.load(f)

    if RECS_PATH.exists():
        with open(RECS_PATH, "r") as f:
            _recommendations = json.load(f)
    else:
        _recommendations = {}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path: Path):
    load_model_and_labels()
    img = load_img(image_path, target_size=_input_size)
    x = img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, 0)

    probs = _model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    label = _class_names[pred_idx] if _class_names and pred_idx < len(_class_names) else str(pred_idx)

    # top-3
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = []
    for i in top3_idx:
        name = _class_names[i] if _class_names and i < len(_class_names) else str(i)
        top3.append({"label": name, "prob": float(probs[i])})

    rec = _recommendations.get(label, {
        "summary": "This is an automated, non-medical result.",
        "actions": [
            "Do not self-diagnose solely from this app.",
            "Consult a dermatologist for a definitive diagnosis.",
            "If the lesion changes, seek urgent care."
        ]
    })
    return {"label": label, "confidence": conf, "top3": top3, "recommendation": rec}

# ------------------ Routes ------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "photo" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["photo"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            prefix = datetime.now().strftime("%Y%m%d-%H%M%S")
            filename = f"{prefix}_{filename}"
            save_path = UPLOAD_FOLDER / filename
            file.save(str(save_path))
            return redirect(url_for("result", filename=filename))
        else:
            flash("Unsupported file type. Please upload a JPG or PNG.")
            return redirect(request.url)
    return render_template("index.html")

@app.route("/result/<filename>")
def result(filename):
    image_path = UPLOAD_FOLDER / filename
    if not image_path.exists():
        flash("File not found. Please upload again.")
        return redirect(url_for("index"))
    output = predict_image(image_path)
    return render_template("result.html", filename=filename, **output)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=False)
import os

if __name__ == "__main__":
    load_model_and_labels()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
