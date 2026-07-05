import sys
import time
import threading
import cv2
from ultralytics import YOLO

# 1. Load your trained model weights
MODEL_PATH = "src/models/yolov11n-seg.pt" 
print("Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Global variables shared between parallel processing threads
current_frame = None
annotated_frame = None
running = True

# =========================================================
# THREADED BACKEND: Continuously captures webcam video frames
# =========================================================
def camera_capture_thread():
    global current_frame, running
    
    # CAP_V4L2 forces OpenCV to bypass slow translation layers on modern Pi OS
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not connect to the Raspberry Pi Camera.")
        running = False
        return

    while running:
        success, frame = cap.read()
        if not success:
            continue
        # Mirror frame horizontally for a more natural mirror reflection appearance
        current_frame = cv2.flip(frame, 1)
        
    cap.release()

# Start the dedicated camera pipeline thread background process
capture_worker = threading.Thread(target=camera_capture_thread, daemon=True)
capture_worker.start()

# Wait briefly for the camera to spin up and feed initial data arrays
print("Waiting for camera thread initialization...")
while current_frame is None and running:
    time.sleep(0.1)

if not running:
    sys.exit()

print("\n🚀 System active! Displaying decoupled fluid video feedback window.")
print("Press 'q' inside the video pop-up screen to terminate.")

prev_time = 0

# =========================================================
# MAIN THREAD: Handles GUI Windows rendering & background AI math
# =========================================================
while running:
    # Always pull the absolute freshest frame from our background thread worker
    frame_to_process = current_frame.copy() if current_frame is not None else None
    
    if frame_to_process is not None:
        # Run inference on the current available snapshot
        results = model.predict(source=frame_to_process, conf=0.5, verbose=False)

        # Plot matching boundaries only if detections are actively registered
        if results and len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame_to_process

        # Calculate processing loop frame rates
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Overlay the processing engine metrics seamlessly onto the layout screen
        cv2.putText(annotated_frame, f"AI Refresh Rate: {int(fps)} FPS", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display the live window feed fluidly without lockups
        cv2.imshow("Raspberry Pi - YOLO Fluid Real-Time Test", annotated_frame)

    # Break loop safely if user taps 'q' on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
print("\nCamera tracking session cleanly completed.")
