from typing import Literal
import numpy as np
import os
from deepface import DeepFace


class DeepFaceService:
    def __init__(
        self,
        db_path: str | None = "face_recognition_db",
        detection_model: Literal[
            "opencv",
            "retinaface",
            "mtcnn",
            "ssd",
            "dlib",
            "mediapipe",
            "yolov8",
            "yolov11n",
            "yolov11s",
            "yolov11m",
            "centerface",
            "skip",
        ] = "opencv",
        normalization_model: Literal[
            "base",
            "raw",
            "Facenet",
            "Facenet2018",
            "VGGFace",
            "VGGFace2",
            "ArcFace",
        ] = "base",
        recognition_model: Literal[
            "VGG-Face",
            "Facenet",
            "Facenet512",
            "OpenFace",
            "DeepFace",
            "DeepID",
            "Dlib",
            "ArcFace",
            "SFace",
            "GhostFaceNet",
        ] = "VGG-Face",
    ):
        """
        Face recognition using DeepFace library.

        Parameters:
            db_path (str, optional): Path to directory that contains subdirectories, where each subdirectory is named after a person and contains images of that person. Default is "face_recognition_db.
            detection_model (string, optional): Face detector backend. Options: opencv, retinaface, mtcnn, ssd, dlib, mediapipe, yolov8, yolov11n, yolov11s, yolov11m, centerface or skip. Default is opencv.
            normalization_model (string, optional): Normalize the input image before feeding it to the model. Options: base, raw, Facenet, Facenet2018, VGGFace, VGGFace2, ArcFace. Default is base.
            recognition_model (string, optional): Model for face recognition. Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, Dlib, ArcFace, SFace and GhostFaceNet Default is VGG-Face.
        """
        self.detection_model = detection_model
        self.normalization_model = normalization_model
        self.recognition_model = recognition_model
        self.known_face_embeddings = []
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
        enforce_detection: bool = False,
        anti_spoofing: bool = False,
        expand_percentage: int = 0,
        align: bool = True,
        max_faces: int | None = None,
    ) -> None:
        """
        Generate embeddings for images in db_path.

        Parameters:
        """
        for name in os.listdir(self.db_path):
            dirname = os.path.join(self.db_path, name)
            for filename in os.listdir(dirname):
                image = os.path.join(dirname, filename)
                embeddings = DeepFace.represent(
                    img_path=image,
                    enforce_detection=enforce_detection,
                    anti_spoofing=anti_spoofing,
                    expand_percentage=expand_percentage,
                    align=align,
                    max_faces=max_faces,
                    detector_backend=self.detection_model,
                    normalization=self.normalization_model,
                    model_name=self.recognition_model,
                )
                print(embeddings)
                if len(embeddings) <= 0:
                    continue
                self.known_face_embeddings.append(embeddings[0]["embedding"])
                self.known_face_names.append(name)

    def find(
        self,
        image,
        tolerance: float = 0.6,
        expand_percentage: int = 0,
        anti_spoofing: bool = False,
        enforce_detection: bool = False,
        align: bool = True,
        max_faces: int | None = None,
    ) -> list[str]:
        """
        Finds faces in an image and compares with known faces to return matched names.

        Parameters:
            image: The image that contains one or more faces.
            tolerance (float, optional): Distance between face comparison to consider as a match. Lower number is more strict. Default is 0.6.
            enforce_detection (boolean, optional): If no face is detected in an image, raise an exception. Default is False.
            align (boolean, optional): Perform alignment based on the eye positions. Default is True.
            expand_percentage (int, optional): Expand detected facial area with a percentage. Default is 0.
            anti_spoofing (boolean, optional): Flag to enable anti spoofing. Default is False.
            max_faces (int, optional): Set a limit on the number of faces to be processed. Default is None.
        Returns:
            list[str]: List of names of faces recognized in the image. Default is empty list.
        """
        if not self.known_face_embeddings or not self.known_face_names:
            return []

        faces = DeepFace.represent(
            img_path=image,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
            expand_percentage=expand_percentage,
            align=align,
            max_faces=max_faces,
            detector_backend=self.detection_model,
            normalization=self.normalization_model,
            model_name=self.recognition_model,
        )
        if len(faces) <= 0:
            return

        names = []

        for face in faces:
            face_embedding = face["embedding"]
            matches = self.compare_faces(
                self.known_face_embeddings,
                face_embedding,
                tolerance=tolerance,
            )
            face_distances = self.face_distance(
                self.known_face_embeddings, face_embedding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                names.append(self.known_face_names[best_match_index])
        return names

    def face_distance(self, face_embeddings, face_embedding):
        """
        Compare list of embeddings to a face embedding and get the euclidean distance for each face embedding. The distance tells you how similar the faces are.

        Parameters:
            face_embeddings (list): List of face embeddings.
            face_embedding: Single face embedding to compare.
        Returns:
            list[np.ndarray]:
                List of numpy ndarray distance for each face in the same order as 'face_embeddings'.
        """
        if len(face_embeddings) == 0:
            return np.empty((0))

        return np.linalg.norm(
            np.array(face_embeddings) - np.array(face_embedding), axis=1
        )

    def compare_faces(self, face_embeddings, face_embedding, tolerance=0.6):
        """
        Compare a list of face embeddings against a candidate embedding to see if they match.

        Parameters:
            face_embeddings (list): List of face embeddings.
            face_embedding: Single face embedding to compare.
            tolerance (float, optional): Distance between face comparison to consider as a match. Lower number is more strict. Default is 0.6.
        Returns:
            list[bool]:
                List of boolean for each face in the same order as 'face_embeddings'.
        """
        return list(self.face_distance(face_embeddings, face_embedding) <= tolerance)
