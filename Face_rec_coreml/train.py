import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import coremltools as ct
import os
import sys
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model and convert it to Core ML.')
    parser.add_argument('--relative_path', required=True, type=str, help='The relative path to the dataset and model save directory.')
    parser.add_argument('--classes', nargs='+', required=True, help='List of class names.')

    args = parser.parse_args()
    relativePath = "uploads/"+args.relative_path
    # Set parameters
    img_size = 224
    batch_size = 32
    classes = args.classes
    coreml_model = relativePath+'/people.mlmodel'
    train_dir = relativePath+'/dataset/train'
    test_dir = relativePath+'/dataset/test'
    num_classes = len(classes)  # Set this according to your number of classes

    # Prepare the data
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    validation_datagen = ImageDataGenerator(rescale=1.0/255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    # Load the MobileNetV2 model pre-trained on ImageNet
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    # Add custom layers on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Combine the base model and the custom layers
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(
        train_generator,
        epochs=5,
        validation_data=validation_generator
    )
    model.save(relativePath+'/people.h5')
    # Convert the model to Core ML
    mlmodel = ct.convert(model, inputs=[ct.ImageType(shape=(1, 224, 224, 3,), scale=1/255)],classifier_config = ct.ClassifierConfig(classes),convert_to="neuralnetwork")

    # Save the Core ML model
    mlmodel.save(coreml_model)