import cv2
import os

def capture_snapshots(output_dir, num_snapshots=20):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    count = 0
    while count < num_snapshots:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam Snapshot', frame)
        
        # Save the frame as an image file
        img_filename = os.path.join(output_dir, f'snapshot_{count+1}.jpg')
        cv2.imwrite(img_filename, frame)
        
        print(f'Snapshot {count+1} saved at {img_filename}')
        
        count += 1
        
        # Wait for 1 second before taking the next snapshot
        cv2.waitKey(1000)  # 1000 milliseconds = 1 second

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Usage
output_directory = 'dataset/train/dhanesh'
capture_snapshots(output_directory, num_snapshots=20)
