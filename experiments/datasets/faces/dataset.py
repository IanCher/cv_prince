from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from PIL import Image

import numpy as np

from experiments.datasets.faces.utils import FaceAnnotation, FaceImage


class FaceLabel(Enum):
    non_face: int = 0
    face: int = 1


@dataclass
class FaceDSParameters:
    seed: int = 12345
    img_size: tuple[int, int] = (24, 24)
    non_face_max_overlap: float = 0.15
    non_face_min_proportion: float = 0.85
    non_faces_num_instances: int = 5
    non_faces_max_trials: int = 100


class FaceDataset:
    """
    A class to handle face vs non-face images for detection.
    Follows the PyTorch Dataset class interface
    """

    def __init__(
        self,
        annotation_dir: Path | str,
        imgs_dir: Path | str,
        params: FaceDSParameters = FaceDSParameters(),
    ):

        self.annotations_dir = annotation_dir
        self.imgs_dir = imgs_dir
        self.params = params
        self.face_images: list[FaceImage] = []
        self.images_to_fold: np.ndarray | None = None
        self.faces_arr: np.ndarray | None = None
        self.non_faces_arr: np.ndarray | None = None
        self.image_to_face_data: dict[int, list[int]] = defaultdict(list)
        self.image_to_non_face_data: dict[int, list[int]] = defaultdict(list)
        self.rng = np.random.default_rng(seed=self.seed)

        self.__read_face_dataset_information()

    def __len__(self) -> int:
        if self.faces_arr is None:
            self.__load()

        return self.num_face_imgs + self.num_non_face_imgs

    def __getitem__(self, idx) -> tuple[np.ndarray, FaceLabel]:
        if idx < self.num_face_imgs:
            return self.faces_arr[idx, :], FaceLabel.face

        return self.faces_arr[idx - self.num_face_imgs, :], FaceLabel.non_face

    @property
    def num_face_imgs(self) -> int:
        if self.faces_arr is None:
            self.__load()

        return self.faces_arr.shape[0]

    @property
    def num_non_face_imgs(self) -> int:
        if self.non_faces_arr is None:
            self.__load()

        return self.non_faces_arr.shape[0]

    def transform_image(self, image: Image.Image) -> np.ndarray:
        image = image.convert("L")
        image = image.resize(self.img_size)
        image = np.asarray(image, dtype=np.float64)
        image = image.flatten()
        image /= 255.0
        return image

    def __read_face_dataset_information(self) -> list[FaceImage]:
        images_to_fold = []
        for fold_ellipse_file in sorted(self.annotations_dir.glob("FDDB-fold-*")):
            if not fold_ellipse_file.name.endswith("ellipseList.txt"):
                continue

            fold_id = int(fold_ellipse_file.name.split("-")[-2])

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
                    face_image = FaceImage(file_path=face_img_path, faces=faces)

                    self.face_images.append(face_image)
                    images_to_fold.append(fold_id)

                self.images_to_fold = np.array(images_to_fold)

    def __load(self):
        print("Loading Face Dataset")

        faces_arr = []
        non_faces_arr = []

        for img_id, face_image in enumerate(self.face_images):
            for face_id in range(face_image.num_faces):
                self.image_to_face_data[img_id].append(len(faces_arr))

                cropped_face = face_image.crop_face_img(face_id)
                faces_arr.append(self.transform_image(cropped_face))

            non_faces = face_image.crop_non_face_img(
                max_overlap=self.non_face_max_overlap,
                min_allowed_size_proportion=self.non_face_min_proportion,
                num_instances=self.non_faces_num_instances,
                max_trials=self.non_faces_max_trials,
                rng=self.rng,
            )

            for non_face in non_faces:
                self.image_to_non_face_data[img_id].append(len(non_faces_arr))
                non_faces_arr.append(self.transform_image(non_face))

        self.faces_arr = np.stack(faces_arr, axis=0)
        self.non_faces_arr = np.stack(non_faces_arr, axis=0)

        print("Face Dataset Loaded")

    @property
    def seed(self) -> int:
        return self.params.seed

    @property
    def img_size(self) -> tuple[int, int]:
        return self.params.img_size

    @property
    def non_face_max_overlap(self) -> float:
        return self.params.non_face_max_overlap

    @property
    def non_face_min_proportion(self) -> float:
        return self.params.non_face_min_proportion

    @property
    def non_faces_num_instances(self) -> int:
        return self.params.non_faces_num_instances

    @property
    def non_faces_max_trials(self) -> int:
        return self.params.non_faces_max_trials
