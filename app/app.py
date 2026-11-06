import os

# Keep TensorFlow's CPU thread usage small on a memory-limited service.
# These must be set before importing TensorFlow.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageFilter, UnidentifiedImageError

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "handwritten_digit_model.keras")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# Limit upload size. Images are also reduced in dimensions below.
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
MAX_IMAGE_DIMENSION = 1024

# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Prediction-only app: skip loading training/optimizer configuration.
model = load_model(MODEL_PATH, compile=False)

print("CNN model loaded successfully!", flush=True)
print("Model input shape:", model.input_shape, flush=True)

# ==================================================
# 3. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")

# ==================================================
# 4. IMAGE PREPROCESSING
# ==================================================

def preprocess_image(image):
    """
    Convert an uploaded image into MNIST-style input:
    a centered 28 x 28 grayscale image, normalized to 0-1.
    """

    # Reduce very large images before creating NumPy/HSV arrays.
    image = image.convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
        Image.Resampling.LANCZOS
    )

    # Detect purple/blue ink.
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    ink_mask = (
        (hue >= 150) &
        (hue <= 230) &
        (saturation > 40) &
        (value < 230)
    )

    digit_array = ink_mask.astype(np.uint8) * 255
    del hsv, hue, saturation, value, ink_mask

    # Fallback for black/white canvas drawings and photos.
    if np.count_nonzero(digit_array) < 10:
        gray_image = image.convert("L")
        gray_image = gray_image.filter(ImageFilter.MedianFilter(size=3))
        gray = np.asarray(gray_image, dtype=np.uint8)

        border = np.concatenate((
            gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]
        ))
        light_background = np.median(border) > 127

        histogram = np.bincount(gray.ravel(), minlength=256)
        total = gray.size
        sum_total = np.dot(np.arange(256), histogram)

        sum_background = 0
        weight_background = 0
        max_variance = 0
        threshold = 127

        for t in range(256):
            weight_background += histogram[t]
            if weight_background == 0:
                continue

            weight_foreground = total - weight_background
            if weight_foreground == 0:
                break

            sum_background += t * histogram[t]
            mean_background = sum_background / weight_background
            mean_foreground = (
                sum_total - sum_background
            ) / weight_foreground

            variance = (
                weight_background *
                weight_foreground *
                (mean_background - mean_foreground) ** 2
            )

            if variance > max_variance:
                max_variance = variance
                threshold = t

        if light_background:
            digit_array = (gray <= threshold).astype(np.uint8) * 255
        else:
            digit_array = (gray > threshold).astype(np.uint8) * 255

        del gray_image, gray, histogram, border

    # Clean noise and crop to the detected digit.
    digit_image = Image.fromarray(digit_array).filter(
        ImageFilter.MedianFilter(size=3)
    )
    bbox = digit_image.getbbox()

    if bbox is None:
        return None

    digit_image = digit_image.crop(bbox)
    digit_image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Place the digit on a black 28 x 28 canvas.
    canvas = Image.new("L", (28, 28), 0)
    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2
    canvas.paste(digit_image, (x, y))

    # Center by center of mass, matching the existing preprocessing approach.
    pixels = np.asarray(canvas, dtype=np.float32)
    total_mass = pixels.sum()

    if total_mass > 0:
        rows, cols = np.indices(pixels.shape)
        center_x = (cols * pixels).sum() / total_mass
        center_y = (rows * pixels).sum() / total_mass

        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))

        canvas = canvas.transform(
            (28, 28),
            Image.Transform.AFFINE,
            (1, 0, -shift_x, 0, 1, -shift_y),
            resample=Image.Resampling.BILINEAR
        )

    # Return shape (1, 28, 28, 1), values in range 0-1.
    image_array = np.asarray(canvas, dtype=np.float32) / 255.0
    return image_array.reshape(1, 28, 28, 1)

# ==================================================
# 5. PREDICTION API
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image received. Please select an image."}), 400

    uploaded_file = request.files["image"]
    if not uploaded_file.filename:
        return jsonify({"error": "Please select an image first."}), 400

    try:
        image_bytes = uploaded_file.read()
        if not image_bytes:
            return jsonify({"error": "The uploaded image is empty."}), 400

        with Image.open(io.BytesIO(image_bytes)) as source_image:
            source_image.load()
            processed_image = preprocess_image(source_image)

        if processed_image is None:
            return jsonify({"error": "No digit detected. Try another image."}), 400

        # Direct single-image inference instead of model.predict().
        predictions = model(processed_image, training=False).numpy()

        predicted_digit = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]) * 100)

        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2)
        }), 200

    except (UnidentifiedImageError, OSError, ValueError):
        app.logger.exception("Invalid image during prediction")
        return jsonify({"error": "Invalid or unsupported image file."}), 400

    except Exception:
        app.logger.exception("Unexpected error during prediction")
        return jsonify({"error": "Prediction failed. Check the Render logs."}), 500

# ==================================================
# 6. HANDLE LARGE UPLOADS
# ==================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 5 MB."
    }), 413

# ==================================================
# 7. LOCAL DEVELOPMENT
# ==================================================

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
