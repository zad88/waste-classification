
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# Project configuration
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "mobilenetv2_best.keras"

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation"
]

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


def load_and_preprocess_image(image_path):
    """Load an image and prepare it for MobileNetV2."""

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)

    # Apply MobileNetV2 preprocessing: values become [-1, 1].
    image_array = preprocess_input(image_array)

    # Add batch dimension.
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image_path, model):
    """Predict the waste category of a new image."""

    image_array = load_and_preprocess_image(image_path)

    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_category = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    # Get the three highest prediction probabilities.
    top_indices = np.argsort(probabilities)[::-1][:3]

    print("\n========== WASTE CLASSIFICATION RESULT ==========")
    print(f"Image: {Path(image_path).name}")
    print(f"Predicted category: {predicted_category}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nTop 3 predictions:")

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {CLASS_NAMES[index]}: "
            f"{probabilities[index] * 100:.2f}%"
        )

    print("=================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict the waste category of a JPG or PNG image."
    )

    parser.add_argument(
        "image",
        type=Path,
        help="Path to the image you want to classify."
    )

    args = parser.parse_args()
    image_path = args.image

    # Validate image path.
    if not image_path.is_file():
        raise SystemExit(f"Error: Image not found: {image_path}")

    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise SystemExit("Error: Only JPG, JPEG and PNG images are supported.")

    # Validate model path.
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Error: Trained model not found: {MODEL_PATH}")

    print("Loading MobileNetV2 model...")

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.")

    predict_image(image_path, model)


if __name__ == "__main__":
    main()
