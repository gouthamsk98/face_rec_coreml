import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
# Set parameters
img_size = 224
batch_size = 32
classes = ['dhanesh','goutham','unknown']
num_classes = len(classes)  # Set this according to your number of classes

# Prepare the data
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
validation_generator = validation_datagen.flow_from_directory(
    'dataset/test',
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical'
)
# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define image size
img_size = 224

# Load and preprocess an input image
def preprocess_frame(frame):
    img = cv2.resize(frame, (img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Rescale image
    return img_array

# Get the class labels from the training data (assuming train_generator is still available)
# If not, you can save the class indices during training and load them here
print(train_generator.class_indices.items())
class_labels = {v: k for k, v in train_generator.class_indices.items()}
print(class_labels)
# # Function to perform inference on a frame
def predict_frame(frame):
    img_array = preprocess_frame(frame)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)
    return class_labels[predicted_class[0]], predictions[0]

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform inference
    predicted_class, prediction_probabilities = predict_frame(frame)
    
    # Display the resulting frame
    cv2.putText(frame, f"Prediction: {predicted_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    for idx, label in class_labels.items():
        cv2.putText(frame, f"{label}: {prediction_probabilities[idx]*100:.2f}%", (10, 60 + idx*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('Face Recognition', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
