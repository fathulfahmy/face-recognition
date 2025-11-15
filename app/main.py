import cv2
from ultralytics import YOLO


def main():
    from services.face_recognition import FaceRecognitionService

    face_recognition_service = FaceRecognitionService()

    # from services.deepface import DeepFaceService

    # face_recognition_service = DeepFaceService()

    source = 0

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    model = YOLO("yolo11n-pose.pt")

    while True:
        success, frame = cap.read()

        if not success:
            continue

        results = model.predict(frame, verbose=False, conf=0.6)
        result = results[0]
        annotated_frame = result.plot(labels=False, masks=False, probs=False)

        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[int(cls)]
            if class_name != "person":
                continue

            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, xyxy)

            cropped_frame = frame[y1:y2, x1:x2]
            names = face_recognition_service.find(cropped_frame, tolerance=0.9)
            if names:
                name = names[0]
            else:
                name = ""

            cv2.rectangle(
                annotated_frame, (x1, y1 - 35), (x2, y1), (255, 0, 0), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                annotated_frame,
                name,
                (x1 + 6, y1 - 6),
                font,
                1.0,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
