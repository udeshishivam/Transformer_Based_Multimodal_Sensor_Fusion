import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import cv2

# Load the trained model
model = load_model('/home/kaushek/TFGrid/runs/weather_image_recognition_2.keras')

# Load and preprocess the image
img_path = '/home/kaushek/TFGrid/Dataset/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151604012404.jpg'  # Replace with your image path
img = image.load_img(img_path, target_size=(228, 228))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict the class
predictions = model.predict(x)
predicted_class = np.argmax(predictions, axis=1)

# Map the predicted class index to the class label
class_labels = ['fog', 'night', 'rain', 'sandstorm', 'snow', 'sunny']  # Replace with your actual class labels
print(f'Predicted weather condition: {class_labels[predicted_class[0]]}')
cv2.imshow(img_path)