import os
import cv2
import time
import sys
import os
from deployment.inference import predict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Real-Time MSI System...")
    print("Press 'q' to exit.")

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            class_id, label, confidence = predict(frame)
        except Exception as e:
            print(f"Inference error: {e}")
            class_id, label, confidence = -1, "Error", 0.0


        if label == "Unknown" or class_id == 6:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        text_label = f"Material: {label}"
        text_conf = f"Conf: {confidence:.2f}"
        
        cv2.putText(frame, text_label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, text_conf, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        prev_frame_time = new_frame_time

        cv2.imshow('Material Stream Identification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()