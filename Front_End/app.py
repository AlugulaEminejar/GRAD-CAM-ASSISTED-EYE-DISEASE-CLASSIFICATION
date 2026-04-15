from flask import Flask, render_template, request
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# ================= FLASK CONFIG =================
app = Flask(__name__)

UPLOAD_ROOT = "static/uploads"
MODEL1_UPLOAD = os.path.join(UPLOAD_ROOT, "model1_retinal_c8")
MODEL2_UPLOAD = os.path.join(UPLOAD_ROOT, "model2_dataset_101")

os.makedirs(MODEL1_UPLOAD, exist_ok=True)
os.makedirs(MODEL2_UPLOAD, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ================= LOAD MODELS =================
retinal_c8_model = load_model("retinal_c8_model.h5", compile=False)
dataset_101_model = load_model("model_1_CNN.keras", compile=False)

DATASET_101_INPUT_SIZE = dataset_101_model.input_shape[1:3]

print("✅ Models loaded")
print("📐 Dataset-101 input size:", DATASET_101_INPUT_SIZE)

# ================= LABELS =================
RETINAL_C8_LABELS = [
    "AMD", "CNV", "CSR", "DME",
    "DR", "DRUSEN", "MH", "NORMAL"
]

DATASET_101_LABELS = [
    "ACRIMA","Glaucoma" ,"ODIR-5K","ORIGA","cataract"
]

# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_retinal_c8(path):
    img = image.load_img(path, target_size=(224, 224))
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0), arr.astype(np.uint8)


def preprocess_dataset_101(path):
    img = image.load_img(path, target_size=DATASET_101_INPUT_SIZE)
    arr = image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ================= GRAD-CAM (MODEL-1 ONLY) =================
def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None


def generate_gradcam(img_arr, model, class_idx):
    last_conv = get_last_conv_layer(model)
    if last_conv is None:
        return None

    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_arr)
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]

    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    if tf.reduce_max(heatmap) == 0:
        return None

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, 0.4, img, 0.6, 0)


# ================= ROUTES =================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dataset")
def dataset():
    return render_template("dataset.html")


@app.route("/instructions")
def instructions():
    return render_template("instructions.html")


# ================= MODEL-1 ROUTE =================
@app.route("/model1", methods=["GET", "POST"])
def model1():
    result = None
    image_path = None
    gradcam_image = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not allowed_file(file.filename):
            return render_template("model1.html", result="❌ Invalid image")

        filename = secure_filename(file.filename)
        path = os.path.join(MODEL1_UPLOAD, filename)
        file.save(path)
        image_path = path

        img_arr, raw_img = preprocess_retinal_c8(path)
        preds = retinal_c8_model.predict(img_arr)[0]

        idx = np.argmax(preds)
        label = RETINAL_C8_LABELS[idx]
        confidence = float(np.max(preds)) * 100

        heatmap = generate_gradcam(img_arr, retinal_c8_model, idx)
        if heatmap is not None:
            overlay = overlay_heatmap(raw_img, heatmap)
            gradcam_path = path.replace(".", "_gradcam.")
            cv2.imwrite(gradcam_path, overlay)
            gradcam_image = gradcam_path

        result = f"{label} ({confidence:.2f}%)"

    return render_template(
        "model1.html",
        result=result,
        image_path=image_path,
        gradcam_image=gradcam_image
    )


# ================= MODEL-2 ROUTE =================
@app.route("/model2", methods=["GET", "POST"])
def model2():
    result = None
    image_path = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not allowed_file(file.filename):
            return render_template("model2.html", result="❌ Invalid image")

        filename = secure_filename(file.filename)
        path = os.path.join(MODEL2_UPLOAD, filename)
        file.save(path)
        image_path = path

        img_arr = preprocess_dataset_101(path)
        preds = dataset_101_model.predict(img_arr)[0]

        idx = np.argmax(preds)
        label = DATASET_101_LABELS[idx]
        confidence = float(np.max(preds)) * 100

        result = f"{label} ({confidence:.2f}%)"

    return render_template(
        "model2.html",
        result=result,
        image_path=image_path
    )


# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
