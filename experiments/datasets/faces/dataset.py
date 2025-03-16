"""
Classes to load and iterate over FDDB dataset
https://people.cs.umass.edu/~elm/papers/fddb.pdf
"""

import itertools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.datasets.faces.utils import (
    FaceAnnotation,
    FaceImage,
    OtherSelectionParams,
)


class FaceLabel(Enum):
    """Labels for face detection"""

    OTHER: int = 0
    FACE: int = 1


@dataclass
class FaceDSParameters:
    """Parameters for manipulating Face Datasets"""

    seed: int = 12345
    img_size: tuple[int, int] = (24, 24)
    other_selection_params = OtherSelectionParams(
        max_overlap=0.15,
        min_proportion=0.85,
        num_instances=2,
        max_trials=100,
    )


@dataclass
class FaceDSDirectories:
    """Directoris to access Face Dataset"""

    annotations: Path
    images: Path


class FaceDataset:
    """
    A class to handle face vs non-face images for detection.
    Follows the PyTorch Dataset class interface
    """

    def __init__(
        self,
        annotations_dir: Path | str,
        imgs_dir: Path | str,
        params: FaceDSParameters = FaceDSParameters(),
    ):

        self.dirs = FaceDSDirectories(annotations=annotations_dir, images=imgs_dir)
        self.params = params
        self.face_images: list[FaceImage] = []
        self.rng = np.random.default_rng(seed=self.seed)

        self.__read_face_dataset_information()
        self.img_name_to_img_id = {
            img.name: idx for idx, img in enumerate(self.face_images)
        }
        self.idx_to_img_area_id = (
            self.__face_idx_to_img_area_id() + self.__other_idx_to_img_area_id()
        )
        self.fold_to_image_names = self.__fold_to_image_names()

    def __len__(self) -> int:
        return sum(map(lambda x: x.num_faces + x.num_others, self.face_images))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, FaceLabel]:
        face_id, area_id = self.idx_to_img_area_id[idx]
        face_img = self.face_images[face_id]

        if idx < self.num_face_imgs:
            return self.transform_image(face_img.crop_face_img(area_id)), FaceLabel.FACE

        return self.transform_image(face_img.crop_other_img(area_id)), FaceLabel.OTHER

    def get_data_from_folds(self, folds: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns all images and corresponding labels contained in the requested folds

        PARAMETERS
        ----------
        folds: list[int]
            A list containing fold indices as defined in FDDB

        RETURNS
        -------
        images: np.ndarray[float]
            All images contained in folds, staked in an array of shape (N, D*D)
        labels: np.ndarray[FaceLabels]
            The labels corresponding to images, shape (N,)
        """
        image_names = itertools.chain.from_iterable(
            [self.fold_to_image_names[fold_id] for fold_id in folds]
        )
        image_names = list(image_names)

        num_faces_in_folds = 0
        num_others_in_folds = 0
        for name in itertools.chain.from_iterable(
            [self.fold_to_image_names[fold_id] for fold_id in folds]
        ):
            face_img = self.get_face_from_name(name)
            num_faces_in_folds += face_img.num_faces
            num_others_in_folds += face_img.num_others

        faces_arr = np.zeros((num_faces_in_folds, np.prod(self.img_size)))
        others_arr = np.zeros((num_others_in_folds, np.prod(self.img_size)))

        face_idx = 0
        other_idx = 0
        for name in image_names:
            face_img = self.get_face_from_name(name)

            for cropped_face in face_img.crop_all_faces_img():
                faces_arr[face_idx, :] = self.transform_image(cropped_face)
                face_idx += 1

            for cropped_other in face_img.crop_all_others_img():
                others_arr[other_idx, :] = self.transform_image(cropped_other)
                other_idx += 1

        data = np.concatenate([faces_arr, others_arr])
        labels = np.array(
            num_faces_in_folds * [FaceLabel.FACE]
            + num_others_in_folds * [FaceLabel.OTHER]
        )

        return data, labels

    def get_face_from_name(self, name: str) -> FaceImage:
        """
        Returns a FaceImage object corresponding to the image name defined in FDDB

        PARAMETERS
        ----------
        name: str
            The image name as defined in FDDB, typically '/2002/07/19/big/img_18'

        RETURNS
        -------
        face_image: FaceImage
            A FaceImage object that allows that contains the exact position of the faces
            and allows to crop face on non face regions
        """
        return self.face_images[self.img_name_to_img_id[name]]

    @property
    def num_face_imgs(self) -> int:
        """The total number of faces that can be extracted from the FDDB"""

        return sum(map(lambda x: x.num_faces, self.face_images))

    @property
    def num_other_imgs(self) -> int:
        """The total number of non faces that can be extracted from the FDDB"""

        return sum(map(lambda x: x.num_others, self.face_images))

    def transform_image(self, image: Image.Image) -> np.ndarray:
        """Some transformations applied to the image before loading them

        PARAMETERS
        ----------
        image: PIL Image
            The image cropped from a FaceImage. Can be a face or other

        RETURNS
        -------
        transformed_image: np.ndarray
            Transformed image, ready for learning and predicting
        """

        image = image.convert("L")
        image = image.resize(self.img_size)
        image = np.asarray(image, dtype=np.float64)
        image = image.flatten()
        image /= 255.0
        return image

    def __read_face_dataset_information(self) -> list[FaceImage]:
        """Read the informations contained in FDDB files to create FaceImage objects"""

        for fold_ellipse_file in sorted(self.annotations_dir.glob("FDDB-fold-*")):
            if not fold_ellipse_file.name.endswith("ellipseList.txt"):
                continue

            with open(fold_ellipse_file, "r", encoding="utf-8") as fid:
                while True:
                    line = fid.readline()

                    if not line:
                        break

                    if line.endswith("\n"):
                        line = line[:-1]

                    assert line.startswith("2002") or line.startswith("2003")

                    face_img_path = self.imgs_dir / (line + ".jpg")
                    num_faces = int(fid.readline())
                    faces = [
                        FaceAnnotation.from_str(fid.readline())
                        for _ in range(num_faces)
                    ]
                    face_image = FaceImage(
                        file_path=face_img_path,
                        face_annotations=faces,
                        other_selection_params=self.other_selection_params,
                        rng=self.rng,
                    )
                    self.face_images.append(face_image)

    def __face_idx_to_img_area_id(self) -> list[tuple[int, int]]:
        return [
            (face_image_id, face_area_id)
            for face_image_id, face_image in enumerate(self.face_images)
            for face_area_id in range(face_image.num_faces)
        ]

    def __other_idx_to_img_area_id(self) -> list[tuple[int, int]]:
        return [
            (face_image_id, other_area_id)
            for face_image_id, face_image in enumerate(self.face_images)
            for other_area_id in range(face_image.num_others)
        ]

    def __fold_to_image_names(self) -> dict[int, list[str]]:
        fold_to_images = {}

        for fold_file in sorted(self.annotations_dir.glob("FDDB-fold-*")):
            if fold_file.name.endswith("ellipseList.txt"):
                continue

            fold_id = int(fold_file.stem.split("-")[-1])

            with open(fold_file, "r", encoding="utf-8") as fid:
                fold_to_images[fold_id] = fid.read().splitlines()

        return fold_to_images

    @property
    def seed(self) -> int:
        """Get the random seed"""

        return self.params.seed

    @property
    def img_size(self) -> tuple[int, int]:
        """Get the size of the crops we extract from FaceImages"""

        return self.params.img_size

    @property
    def other_selection_params(self):
        """Access all the parameters used for cropping non face areas"""

        return self.params.other_selection_params

    @property
    def num_folds(self) -> int:
        """Number of folds predefined in FDDB"""
        return len(self.fold_to_image_names)

    @property
    def imgs_dir(self) -> Path:
        """Access the image directory of FDDB"""
        return self.dirs.images

    @property
    def annotations_dir(self) -> Path:
        """Access the annotations directory of FDDB"""
        return self.dirs.annotations
