"""Detects the 4 colored gripper markers to find position, orientation, open/close state."""

import cv2
import yaml
import math
import numpy as np
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


@dataclass
class GripperState:
    center_px: Tuple[int, int]
    orientation: float
    is_open: bool
    confidence: float
    raw_points: Dict[str, Tuple[int, int]] = field(default_factory=dict)


class GripperDetector:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "color_config.yaml"
            
        self.config_path = Path(config_path)
        self.colors = {}
        self.history = deque(maxlen=5)  # smooth center over frames
        self.load_config()

    def load_config(self):
        self.colors = {
            "outer_left": {    
                "lower": np.array([25, 40, 40]),
                "upper": np.array([55, 255, 255])
            },
            "outer_right": {   
                "lower": np.array([145, 50, 40]), 
                "upper": np.array([175, 255, 255])
            },
            "inner_left": {    
                "lower": np.array([5, 70, 50]),
                "upper": np.array([20, 255, 255])
            },
            "inner_right": { 
                "lower": np.array([125, 40, 40]),
                "upper": np.array([150, 255, 255])
            }
        }

    def detect(self, frame: np.ndarray) -> Optional[GripperState]:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # find center of each colored marker
        points = {}
        for color_name, bounds in self.colors.items():
            center = self._find_color_center(hsv_frame, bounds["lower"], bounds["upper"])
            if center is not None:
                points[color_name] = center

        # need all 4 markers visible
        if len(points) < 4:
            return None

        out_l = points["outer_left"]
        out_r = points["outer_right"]
        in_l = points["inner_left"]
        in_r = points["inner_right"]

        dist_outer = self._distance(out_l, out_r)
        dist_inner = self._distance(in_l, in_r)

        # outer points must be wider than inner points
        if dist_outer <= dist_inner:
            return None

        # geometric center of the gripper
        cx = int((out_l[0] + out_r[0] + in_l[0] + in_r[0]) / 4)
        cy = int((out_l[1] + out_r[1] + in_l[1] + in_r[1]) / 4)

        self.history.append((cx, cy))
        smooth_cx = int(sum(p[0] for p in self.history) / len(self.history))
        smooth_cy = int(sum(p[1] for p in self.history) / len(self.history))

        # open if inner markers are farther apart than this threshold
        is_open_threshold = 50.0
        is_open = dist_inner > is_open_threshold

        # angle of line connecting outer left and outer right
        dx = out_r[0] - out_l[0]
        dy = out_r[1] - out_l[1]
        angle = math.degrees(math.atan2(dy, dx))

        return GripperState(
            center_px=(smooth_cx, smooth_cy),
            orientation=angle,
            is_open=is_open,
            confidence=1.0,
            raw_points=points
        )

    def _find_color_center(self, hsv: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Optional[Tuple[int, int]]:
        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_center = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if 15 < area < 1500:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = max(w / float(h), h / float(w))
                
                if aspect_ratio < 2.5:
                    if area > max_area:  
                        M = cv2.moments(cnt)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            best_center = (cx, cy)
                            max_area = area

        return best_center

    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = GripperDetector()
    print("Testing Gripper Detector in Debug Mode. Press 'q' to quit.")
    
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
            cv2.putText(frame, f"Gripper Target {state.is_open}", (cx - 50, cy - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Open: {state.is_open}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Waiting for all 4 colors...", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        cv2.imshow("Gripper Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
