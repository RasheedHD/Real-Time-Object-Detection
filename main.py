import cv2
from ultralytics import YOLO

def main():
    # Load pretrained YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run object detection
        results = model(frame)

        # Draw results on frame
        annotated_frame = results[0].plot()

        # Show the frame
        cv2.imshow("YOLOv8 Real-Time Object Detection", annotated_frame)

        # Quit on pressing q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()