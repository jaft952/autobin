"""
src/perception/color_calibrator.py

Utility to tune HSV bounds for the 4 gripper colors.
Values are saved to config/color_config.yaml.
"""

import cv2
import yaml
import numpy as np
from pathlib import Path


def nothing(x):
    pass


def main():
    config_path = Path(__file__).parent / "config" / "color_config.yaml"
    
    # Load existing config or create default
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print("Config file not found, using defaults.")
        return

    color_keys = ["outer_left", "outer_right", "inner_left", "inner_right"]
    
    cv2.namedWindow("Calibrator")
    cv2.createTrackbar("Color Select", "Calibrator", 0, 3, nothing)
    
    # HSV Trackbars
    cv2.createTrackbar("H Min", "Calibrator", 0, 179, nothing)
    cv2.createTrackbar("S Min", "Calibrator", 0, 255, nothing)
    cv2.createTrackbar("V Min", "Calibrator", 0, 255, nothing)
    cv2.createTrackbar("H Max", "Calibrator", 179, 179, nothing)
    cv2.createTrackbar("S Max", "Calibrator", 255, 255, nothing)
    cv2.createTrackbar("V Max", "Calibrator", 255, 255, nothing)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for i in range(30):
        cap.read() 
    last_color_idx = -1

    print("Starting calibrator. Press 's' to save. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        color_idx = cv2.getTrackbarPos("Color Select", "Calibrator")
        current_color = color_keys[color_idx]

        # Update trackbars if color selection changed
        if color_idx != last_color_idx:
            lower = config["colors"][current_color]["lower"]
            upper = config["colors"][current_color]["upper"]
            cv2.setTrackbarPos("H Min", "Calibrator", lower[0])
            cv2.setTrackbarPos("S Min", "Calibrator", lower[1])
            cv2.setTrackbarPos("V Min", "Calibrator", lower[2])
            cv2.setTrackbarPos("H Max", "Calibrator", upper[0])
            cv2.setTrackbarPos("S Max", "Calibrator", upper[1])
            cv2.setTrackbarPos("V Max", "Calibrator", upper[2])
            last_color_idx = color_idx

        # Read current trackbar values
        h_min = cv2.getTrackbarPos("H Min", "Calibrator")
        s_min = cv2.getTrackbarPos("S Min", "Calibrator")
        v_min = cv2.getTrackbarPos("V Min", "Calibrator")
        h_max = cv2.getTrackbarPos("H Max", "Calibrator")
        s_max = cv2.getTrackbarPos("S Max", "Calibrator")
        v_max = cv2.getTrackbarPos("V Max", "Calibrator")

        # Update config dictionary
        config["colors"][current_color]["lower"] = [h_min, s_min, v_min]
        config["colors"][current_color]["upper"] = [h_max, s_max, v_max]

        # Convert to HSV and apply mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Show label on frame
        cv2.putText(frame, f"Tuning: {current_color}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Original", frame)
        cv2.imshow("Calibrator", result)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            print(f"Saved configuration to {config_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()