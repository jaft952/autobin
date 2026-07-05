import sys
import time
import cv2
from ultralytics import YOLO

# 1. Load your trained model weights
MODEL_PATH = "src/models/yolov11n-seg.pt"
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# 2. Direct OpenCV Video Capture for Modern Pi OS (Bookworm/Trixie)
print("Initializing camera pipeline...")
# CAP_V4L2 forces OpenCV to use the standard Linux video driver directly
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set the camera resolution to 640x480 to keep the frame rate fast on the Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Error: Could not connect to the Raspberry Pi Camera.")
    print("Make sure your camera is enabled or try running 'rpicam-hello' to test it.")
    sys.exit()

print("\n🚀 Camera successfully connected!")
print("Displaying real-time pop-up window. Press 'q' inside the window to close.")

prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Error: Failed to grab video frame.")
        break
    
    # Mirror the frame horizontally for a natural layout
    frame = cv2.flip(frame, 1)
    annotated_frame = frame

    # Run inference (verbose=False saves a lot of CPU power)
    results = model.predict(source=frame, conf=0.5, verbose=False)

    if results and len(results) > 0:
        annotated_frame = results[0].plot()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Standard real-time GUI desktop pop-up window
    cv2.imshow("Raspberry Pi - YOLO Real-Time Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nCamera feed shut down successfully.")
