import cv2 as cv
from ultralytics import YOLO


cap = cv.VideoCapture(0)
model = YOLO("yolo26n.pt")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, tracker="deepocsort.yaml",)

    annotated_frame = results[0].plot()

    cv.imshow("YOLO Object Detection", annotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv.destroyAllWindows()