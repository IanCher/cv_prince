"""Scripts to run face detection experiments"""

from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
from cv_prince.chap_07_complex_densities.gmm import (
    ExpectationMaximisationGMM,
    GMMSampler,
)
from experiments.datasets.faces import FaceLabel


@dataclass
class GMMFaceDetectorParams:
    """Parameters for the face detector"""

    seed: int = 12345
    ncomponents_faces: int = 10
    ncomponents_others: int = 10
    em_max_iter: int = 1000
    thresh: float = 0.5


class GMMFaceDetector:
    """Trains and infers face detection using GMM estimations"""

    def __init__(self, params: GMMFaceDetectorParams = GMMFaceDetectorParams):
        self.params = params
        self.em_estimators = {
            FaceLabel.other: ExpectationMaximisationGMM(
                num_components=self.ncomponents_others, seed=self.seed
            ),
            FaceLabel.face: ExpectationMaximisationGMM(
                num_components=self.ncomponents_faces, seed=self.seed
            ),
        }

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """Fit a GMM for faces and for other images"""

        for label in FaceLabel:
            data_with_label = data[labels == label, :]
            self.em_estimators[label].fit(data_with_label, max_iter=self.max_iter)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict if the image represents a face or not"""

        faces_score = self.score(data)

        return np.where(faces_score > self.thresh, FaceLabel.face, FaceLabel.other)

    def score(self, data: np.ndarray) -> np.ndarray:
        """Compute the score of the data representing a face"""

        faces_log_likelihood = self.gmm_faces.log_pdf(data)
        others_log_likelihood = self.gmm_others.log_pdf(data)

        max_likelihood = np.maximum(faces_log_likelihood, others_log_likelihood)
        lse = max_likelihood + np.log(
            np.exp(faces_log_likelihood - max_likelihood)
            + np.exp(others_log_likelihood - max_likelihood)
        )

        return np.exp(faces_log_likelihood - lse)

    def save(self, filepath: Path):
        """Saves model in filepath"""
        if not filepath.parent.exists():
            filepath.parent.mkdir(parents=True)

        with open(filepath, "wb") as fid:
            pickle.dump(self.__dict__, fid)

    def load(self, filepath: Path):
        """Loads model from filepath"""

        with open(filepath, "rb") as fid:
            data = pickle.load(fid)

        self.__dict__.update(data)

    @property
    def seed(self) -> int:
        """Access random seed from params"""

        return self.params.seed

    @property
    def thresh(self) -> float:
        """Access prediction threshold from params"""

        return self.params.thresh

    @thresh.setter
    def thresh(self, val):
        """Set the prediction threshold"""

        self.params.thresh = val

    @property
    def ncomponents_faces(self) -> int:
        """Access ncomponents_faces from params"""

        return self.params.ncomponents_faces

    @property
    def ncomponents_others(self) -> int:
        """Access ncomponents_others from params"""

        return self.params.ncomponents_others

    @property
    def max_iter(self) -> int:
        """Access max_iter from params"""

        return self.params.em_max_iter

    @property
    def gmm_faces(self) -> GMMSampler:
        """Access gmm estimated for the faces"""

        return self.em_estimators[FaceLabel.face].gmm

    @property
    def gmm_others(self) -> GMMSampler:
        """Access gmm estimated for the others"""

        return self.em_estimators[FaceLabel.other].gmm
