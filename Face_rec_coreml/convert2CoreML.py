import coremltools as ct
from tensorflow.keras.models import load_model

# Set parameters
img_size = 224
loaded_model = 'face_recognition_mobilenet.h5'
final_model = 'people.mlmodel'
classes = ['dhanesh', 'naresh']


# Load the trained Keras model
keras_model = load_model(loaded_model)
# keras_model.save('face_recognition_mobilenet_savedmodel', save_format='.keras')
# classifier_config = ct.ClassifierConfig(class_labels)
# Convert the model to Core ML
mlmodel = ct.convert(keras_model,source='tensorflow', inputs=[ct.ImageType(shape=(1, img_size, img_size, 3))])

# Save the Core ML model
mlmodel.save(final_model)