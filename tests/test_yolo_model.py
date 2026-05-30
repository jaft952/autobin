import sys
from pathlib import Path
import cv2
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.perception.detector import AluminiumCanDetector

def main():
    print("Initializing YOLO detector...")
    root_dir = Path(__file__).resolve().parent.parent
    model_path_abs = str(root_dir / "ai/models/subsystem2/production/aluminum_can_detector_best.pt")
    
    detector = AluminiumCanDetector(
        model_path=model_path_abs,
        camera_index=1,  
        conf_threshold=0.80
    )
    try:
        detector.start()
        print("\n[Test Started] Press 'q' to quit.")
        
        last_time = time.time()
        
        while True:
            result = detector.detect()
            
            current_time = time.time()
            fps = 1 / (current_time - last_time) if current_time > last_time else 0
            last_time = current_time
            
            frame = detector.get_annotated_frame(result)
            if frame is None:
                continue
                
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            cv2.imshow("Aluminum Can Detection Test", frame)

            if result.found:
                best_center = result.normalized_center()
                print(f"Can detected at normalized coords: {best_center}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error during testing: {e}")
        
    finally:
        print("\nShutting down...")
        detector.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()