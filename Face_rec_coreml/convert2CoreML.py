import coremltools as ct
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Set parameters
img_size = 224
batch_size = 32
class_labels = [0,1,2]

loaded_model = './face_recognition_mobilenet.h5'
final_model = '~/people3.mlmodel'

# Load the trained Keras model
keras_model = load_model(loaded_model)
# keras_model.save('face_recognition_mobilenet_savedmodel', save_format='.keras')
classifier_config = ct.ClassifierConfig(class_labels)
# Convert the model to Core ML
mlmodel = ct.convert(keras_model,source='tensorflow', inputs=[ct.ImageType(shape=(1, img_size, img_size, 3))],convert_to="neuralnetwork",classifier_config=classifier_config)

# Save the Core ML model
mlmodel.save(final_model)