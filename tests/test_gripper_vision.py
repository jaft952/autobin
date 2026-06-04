import sys
import cv2
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.perception.gripper_detector import GripperDetector

def test_vision():
    cap = cv2.VideoCapture(0)
    detector = GripperDetector()
    print("starting gripper vision test. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for color_name, bounds in detector.colors.items():

            center = detector._find_color_center(hsv, bounds["lower"], bounds["upper"])
            if center:

                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                cv2.putText(frame, color_name, (center[0] + 10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        state = detector.detect(frame)
        if state:
            cx, cy = state.center_px
            cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"Gripper Target", (cx - 50, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:

            cv2.putText(frame, "Waiting for all 4 colors...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Gripper Vision Debug", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_vision()