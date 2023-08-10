import cv2
import numpy as np
import tensorflow as tf

# Load your trained fruit classification model
fruit_classification_model = tf.keras.models.load_model('fruit_classification_model.h5')

# Define class names
class_names = ['apple', 'banana']  # List of class names

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for classification
    resized_frame = cv2.resize(frame, (224, 224))
    input_frame = np.expand_dims(resized_frame, axis=0)
    input_frame = input_frame / 255.0

    # Predict the fruit type
    prediction = fruit_classification_model.predict(input_frame)
    predicted_class_index = int(np.round(prediction[0][0]))
    predicted_class = class_names[predicted_class_index]

    # Draw bounding box and label on the frame
    label = f'{predicted_class} ({prediction[0][0]:.2f})'
    color = (0, 255, 0) if predicted_class == 'apple' else (0, 0, 255)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # Display the frame
    cv2.imshow('Fruit Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera
cap.release()
cv2.destroyAllWindows()
