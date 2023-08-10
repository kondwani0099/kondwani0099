# Run this code if you have trained the model or use my model in the transfer learning folder
import cv2
import serial
import tensorflow as tf
import numpy as np

# Open communication with Arduino
arduino = serial.Serial('COM5', 9600)  # Change the port and baud rate

# Load the trained classification model
model = tf.keras.models.load_model('fruit_classification_model.h5')

# Define class names
class_names = ['apple', 'banana']

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
    prediction = model.predict(input_frame)
    predicted_class_index = int(np.round(prediction[0][0]))
    predicted_class = class_names[predicted_class_index]

    # Send commands to Arduino based on the detected fruit
    if predicted_class == 'apple':
        arduino.write(b'a\n')
    elif predicted_class == 'banana':
        arduino.write(b'b\n')

    # Display the predicted fruit type on the frame
    cv2.putText(frame, f'Predicted: {predicted_class}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Fruit Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close the camera and communication
cap.release()
cv2.destroyAllWindows()
arduino.close()
