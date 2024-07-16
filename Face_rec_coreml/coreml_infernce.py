import cv2
from PIL import Image
import coremltools as ct
import numpy as np

# Load the Core ML model
model = ct.models.MLModel('./people.mlmodel')

def preprocess_image(frame):
    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Resize to match model's input size
    
    image = image.resize((224, 224))
    image=image.rotate(180)
    return image

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Preprocess the frame
        image = preprocess_image(frame)
        
        # Predict using the Core ML model
        predictions = model.predict({'input_1': image})
        
        # Print predictions
        print(predictions)

        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
