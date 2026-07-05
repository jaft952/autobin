import cv2
from ultralytics import YOLO

# Load the trained model weights
# Ensure 'best.pt' is in the specified relative path or move it to your current folder
MODEL_PATH = "../src/models/yolov11n-seg.pt"
model = YOLO(MODEL_PATH) 

# Initialize the webcam stream (0 is usually the default built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam successfully connected. Press 'q' on the video window to exit.")

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Error: Failed to grab frame.")
        break
    
    # Mirror the frame horizontally for a more natural preview appearance
    frame = cv2.flip(frame, 1)
    annotated_frame = frame

    # Run inference using the YOLO model
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # If objects are detected, plot the bounding boxes/segmentation masks
    if results and len(results) > 0:
        annotated_frame = results[0].plot()

    # Display the live window feed
    cv2.imshow("YOLO11 Real-Time Camera Test", annotated_frame)

    # Break the loop immediately if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Safely close down camera resources and windows
cap.release()
cv2.destroyAllWindows()
print("Camera stream closed successfully.")
