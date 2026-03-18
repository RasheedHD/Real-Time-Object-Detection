import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.imshow("Custom YOLO Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()