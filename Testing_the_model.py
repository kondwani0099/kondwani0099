import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('fruit_classification_model.h5')

# model = tf.keras.models.load_model('final_model.h5')

# List of class names (assuming binary classification: apple and banana)
class_names = ['apple', 'banana']

# Load and preprocess an image for testing
img_path = r'C:\Users\TAONGA-PATRICIA\Desktop\Fruit Model\Dataset\test\Banana_3.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values

# Get the model's prediction
prediction = model.predict(img_array)
predicted_class = class_names[int(prediction[0][0])]

print(f"Predicted class: {predicted_class}")
