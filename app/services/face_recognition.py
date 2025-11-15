from typing import Literal
import cv2
import numpy as np
import os
import face_recognition


class FaceRecognitionService:
    def __init__(
        self,
        db_path: str | None = "face_recognition_db",
        detection_model: Literal[
            "hog",
            "cnn",
        ] = "hog",
        encoding_model: Literal[
            "small",
            "large",
        ] = "small",
    ):
        """
        Face recognition using face_recognition library.

        Parameters:
            db_path (str, optional): Path to directory that contains subdirectories, where each subdirectory is named after a person and contains images of that person. Default is "face_recognition_db.
            detection_model (str, optional): Face detection model. "hog" is faster but less accurate (CPU). "cnn" is slower but more accurate (GPU). Default is "hog".
            encoding_model (str, optional): Face encoding model. "small" is faster and returns 5 points. "large" is slower and returns 68 points. Default is "small".
        """
        self.detection_model = detection_model
        self.encoding_model = encoding_model
        self.known_face_encodings = []
        self.known_face_names = []
        self.db_path = (
            os.path.join(
                *db_path.split("/"),
            )
            if db_path
            else None
        )
        if self.db_path:
            self.set_known_faces()

    def set_known_faces(
        self,
        detection_upsample: int = 1,
        encoding_resample: int = 1,
    ) -> None:
        """
        Generate encodings for images in db_path.

        Parameters:
            detection_upsample (int, optional): Number of times to upsample the image for face detection. Higher number find smaller faces. Default is 1.
            encoding_resample (int, optional): Number of times to re-sample the image when encoding. Higher number is slower but more accurate. (100 is 100x slower). Default is 1.
        """
        for name in os.listdir(self.db_path):
            dirname = os.path.join(self.db_path, name)
            for filename in os.listdir(dirname):
                path = os.path.join(dirname, filename)
                rgb_frame = face_recognition.load_image_file(path)
                face_locations = face_recognition.face_locations(
                    img=rgb_frame,
                    number_of_times_to_upsample=detection_upsample,
                    model=self.detection_model,
                )
                encodings = face_recognition.face_encodings(
                    face_image=rgb_frame,
                    known_face_locations=face_locations,
                    num_jitters=encoding_resample,
                    model=self.encoding_model,
                )
                if len(encodings) <= 0:
                    continue
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(name)

    def find(
        self,
        image,
        tolerance: float = 0.6,
        detection_upsample: int = 1,
        encoding_resample: int = 1,
    ) -> list[str]:
        """
        Finds faces in an image and compares with known faces to return matched names.

        Parameters:
            image: The image that contains one or more faces
            tolerance (float, optional): Distance between face comparison to consider as a match. Lower number is more strict. Default is 0.6.
            detection_upsample (int, optional): Number of times to upsample the image for face detection. Higher number find smaller faces. Default is 1.
            encoding_resample (int, optional): Number of times to re-sample the image when encoding. Higher number is slower but more accurate. (100 is 100x slower). Default is 1.
        Returns:
            list[str]: List of names of faces recognized in the image. Default is empty list.
        """
        if not self.known_face_encodings or not self.known_face_names:
            return []

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(
            img=rgb_frame,
            number_of_times_to_upsample=detection_upsample,
            model=self.detection_model,
        )
        face_encodings = face_recognition.face_encodings(
            face_image=rgb_frame,
            known_face_locations=face_locations,
            num_jitters=encoding_resample,
            model=self.encoding_model,
        )
        if len(face_encodings) <= 0:
            return

        names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=tolerance
            )
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                names.append(self.known_face_names[best_match_index])
        return names
