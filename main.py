import cv2 as cv
import pandas as pd

from deepface import DeepFace


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Failed to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to receive frame. Exiting...")
            break

        try:
            faces = DeepFace.find(
                img_path=frame,
                db_path="database/faces",
                enforce_detection=False,
                anti_spoofing=True,
            )
        except Exception:
            faces = []

        if isinstance(faces, list):
            data_frames = faces
        if isinstance(faces, pd.DataFrame):
            data_frames = [faces]

        for data_frame in data_frames:
            if data_frame.empty:
                continue

            for _, row in data_frame.iterrows():
                x = int(row["source_x"])
                y = int(row["source_y"])
                w = int(row["source_w"])
                h = int(row["source_h"])

                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                name = row["identity"].split("/")[-2]
                cv.putText(
                    frame,
                    name,
                    (x, y - 10),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

        cv.imshow("Face Recognition", frame)

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
