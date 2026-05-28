import tensorflow as tf
import numpy as np
from pathlib import Path

# Load the trained model (first save it in your training script)
model = tf.keras.models.load_model('bird_sheep_model.keras')

class_names = ["bird", "sheep"]
img_size = (64, 64)

def predict_image(image_path):
    # Load and preprocess the image
    img = tf.keras.utils.load_img(image_path, target_size=img_size)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0) 
    
    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    print(f"Image: {image_path}")
    print(f"This image is most likely a {predicted_class} with {confidence:.2f}% confidence.")
    print()

test_images = [
    "bird_test.jpg"
    ]

for img_path in test_images:
    if Path(img_path).exists():
        predict_image(img_path)