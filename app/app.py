
import os
import io

import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "handwritten_digit_model.keras"
)

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

# Limit uploaded images to 10 MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


# ==================================================
# 2. LOAD TRAINED MODEL
# ==================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

model = load_model(MODEL_PATH)

print("CNN model loaded successfully!")
print("Model input shape:", model.input_shape)


# ==================================================
# 3. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==================================================
# 4. IMAGE PREPROCESSING
# ==================================================

# def preprocess_image(image):
#     """
#     Convert an input image into the format expected
#     by the trained MNIST CNN model.

#     Expected output shape: (1, 28, 28, 1)
#     """

#     # Convert to grayscale
#     image = image.convert("L")

#     # Reduce small image noise
#     image = image.filter(
#         ImageFilter.MedianFilter(size=3)
#     )

#     # Estimate background brightness from the borders
#     grayscale = np.asarray(image)

#     border_pixels = np.concatenate([
#         grayscale[0, :],
#         grayscale[-1, :],
#         grayscale[:, 0],
#         grayscale[:, -1]
#     ])

#     # MNIST convention: black background, white digit
#     if np.median(border_pixels) > 127:
#         image = ImageOps.invert(image)

#     # Convert to black background and white foreground
#     image = image.point(
#         lambda pixel: 255 if pixel > 80 else 0
#     )

#     # Find the digit boundaries
#     bbox = image.getbbox()

#     if bbox is None:
#         return None

#     # Crop around the digit
#     image = image.crop(bbox)

#     # Resize proportionally, leaving some margin
#     image.thumbnail(
#         (20, 20),
#         Image.Resampling.LANCZOS
#     )

#     # Create a 28 x 28 black image
#     canvas = Image.new(
#         "L",
#         (28, 28),
#         color=0
#     )

#     # Center the digit
#     x = (28 - image.width) // 2
#     y = (28 - image.height) // 2

#     canvas.paste(image, (x, y))

#     # Normalize pixels to the range 0–1
#     image_array = np.asarray(
#         canvas,
#         dtype=np.float32
#     ) / 255.0

#     # Expected CNN input:
#     # (batch_size, height, width, channels)
#     image_array = image_array.reshape(
#         1, 28, 28, 1
#     )

#     return image_array


def preprocess_image(image):
    """
    Preprocess uploaded photos using purple/blue ink detection.
    If no coloured ink is detected, use grayscale thresholding
    as a fallback for black-and-white canvas drawings.

    Output shape: (1, 28, 28, 1)
    """

    # Step 1: Detect purple/blue ink from the original colour image
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    hsv_image = image.convert("HSV")
    hsv = np.asarray(hsv_image, dtype=np.uint8)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Purple/blue pen colour mask
    ink_mask = (
        (hue >= 150) &
        (hue <= 230) &
        (saturation > 40) &
        (value < 230)
    )

    digit_array = ink_mask.astype(np.uint8) * 255

    # Step 2: If coloured ink was not detected, use grayscale fallback
    if np.count_nonzero(digit_array) < 10:

        gray_image = image.convert("L")
        gray_image = gray_image.filter(
            ImageFilter.MedianFilter(size=3)
        )

        gray = np.asarray(gray_image, dtype=np.uint8)

        border = np.concatenate([
            gray[0, :],
            gray[-1, :],
            gray[:, 0],
            gray[:, -1]
        ])

        light_background = np.median(border) > 127

        histogram = np.bincount(
            gray.ravel(), minlength=256
        )

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

            mean_background = (
                sum_background / weight_background
            )

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
            # Include pixels equal to the Otsu threshold.
            # For clean black-on-white canvas images, Otsu can return 0.
            # Using '<' would then detect no black digit pixels at all.
            digit_array = (
                gray <= threshold
            ).astype(np.uint8) * 255
        else:
            digit_array = (
                gray > threshold
            ).astype(np.uint8) * 255

    # Step 3: Remove small isolated noise
    digit_image = Image.fromarray(digit_array)
    digit_image = digit_image.filter(
        ImageFilter.MedianFilter(size=3)
    )

    # Step 4: Crop around detected ink
    bbox = digit_image.getbbox()

    if bbox is None:
        return None

    digit_image = digit_image.crop(bbox)

    # Step 5: Resize proportionally to fit within 20 x 20
    digit_image.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # Step 6: Place on a black 28 x 28 canvas
    canvas = Image.new("L", (28, 28), 0)

    x = (28 - digit_image.width) // 2
    y = (28 - digit_image.height) // 2

    canvas.paste(digit_image, (x, y))

    # Step 7: Centre the digit using its centre of mass
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

    # Step 8: Save debug image
    canvas.save("debug_preprocessed.png")

    # Step 9: Normalize like the MNIST training data
    image_array = np.asarray(
        canvas,
        dtype=np.float32
    ) / 255.0

    return image_array.reshape(1, 28, 28, 1)



# ==================================================
# 5. PREDICTION API
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check that an image was submitted
    if "image" not in request.files:
        return jsonify({
            "error": "No image received. Please select an image."
        }), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({
            "error": "Please select an image first."
        }), 400

    try:
        # Read the uploaded image
        image_bytes = uploaded_file.read()

        if not image_bytes:
            return jsonify({
                "error": "The uploaded image is empty."
            }), 400

        # Open image using Pillow
        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Decode the image before preprocessing
        image.load()

        # Preprocess for the CNN
        processed_image = preprocess_image(image)

        if processed_image is None:
            return jsonify({
                "error": "No digit detected. Try another image."
            }), 400

        # Generate predictions
        predictions = model.predict(
            processed_image,
            verbose=0
        )

        # Select the class with the highest score
        predicted_digit = int(
            np.argmax(predictions[0])
        )

        confidence = float(
            np.max(predictions[0]) * 100
        )

        # Return JSON expected by index.html
        return jsonify({
            "predicted_digit": predicted_digit,
            "confidence": round(confidence, 2)
        }), 200

    except (UnidentifiedImageError, OSError):
        return jsonify({
            "error": "Invalid or unsupported image file."
        }), 400

    except Exception:
        app.logger.exception(
            "Unexpected error during prediction"
        )

        return jsonify({
            "error": "Prediction failed. Check the Flask terminal."
        }), 500


# ==================================================
# 6. HANDLE LARGE UPLOADS
# ==================================================

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": "Image is too large. Maximum size is 10 MB."
    }), 413


# ==================================================
# 7. START FLASK SERVER
# ==================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
