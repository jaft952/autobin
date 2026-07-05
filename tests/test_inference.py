import sys
import time
import cv2
from ultralytics import YOLO

# 1. Load your trained model weights
# Ensure 'best.pt' is in this folder or update this path
MODEL_PATH = "src/models/yolov11n-seg.pt" 
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# 2. Initialize Raspberry Pi Camera Pipeline
try:
    from picamera import PiCamera
    # Check for legacy Pi Camera module compatibility
    is_libcamera = False
except ImportError:
    # Modern Raspberry Pi OS (Bookworm/Bullseye) uses libcamera backend
    from preview_window import PreviewWindow  # Just an alternative check
    is_libcamera = True

if is_libcamera:
    print("Detected modern Raspberry Pi OS. Initializing libcamera pipeline...")
    # On Raspberry Pi OS Bookworm, we must force the v4l2 backend for OpenCV compatibility
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # Set lower resolution to maximize frames-per-second (FPS) on Pi hardware
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
else:
    print("Detected legacy Pi camera system. Initializing default camera...")
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not connect to the Raspberry Pi Camera.")
    print("Verify your ribbon cable connection or run 'libcamera-hello' in the terminal to test hardware.")
    sys.exit()

print("\n🚀 Camera successfully connected!")
print("Displaying real-time pop-up window. Press 'q' inside the window to close.")

# Variables to track frame rate performance
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Error: Failed to grab video frame.")
        break
    
    # Mirror the frame horizontally for a natural layout (optional)
    frame = cv2.flip(frame, 1)
    annotated_frame = frame

    # Run optimized inference (verbose=False keeps terminal clean to save CPU)
    results = model.predict(source=frame, conf=0.5, verbose=False)

    # Plot bounding boxes and segmentation masks if targets exist
    if results and len(results) > 0:
        annotated_frame = results[0].plot()

    # Calculate and overlay current FPS onto the image window
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Generate the standard OS desktop pop-up window
    cv2.imshow("Raspberry Pi - YOLO Real-Time Test", annotated_frame)

    # Break loop safely if user presses the 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources safely
cap.release()
cv2.destroyAllWindows()
print("\nCamera feed shut down successfully.")
