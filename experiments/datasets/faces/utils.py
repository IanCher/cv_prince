"""Classes and functions to manipulate face dataset"""

from dataclasses import astuple, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.geometry.rectangles import Rectangle


@dataclass
class OtherSelectionParams:
    """Parameters to define how we crop non face areas in FaceImage"""

    max_overlap: float = 0.15
    min_proportion: float = 0.8
    num_instances: int | None = None
    max_trials: int = 100


@dataclass
class FaceAnnotation:
    """Face Annotation parameters"""

    major_ax_radius: float
    minor_ax_radius: float
    angle: float
    center_x: float
    center_y: float

    @classmethod
    def from_str(cls, datastr: str):
        """Create a FaceAnnotation from data read in FDDB annotation files"""

        anno = [float(coord) for coord in datastr.split()[:-1]]
        return cls(*anno)

    @cached_property
    def rot_mtx(self) -> np.ndarray:
        """Convert angles in rotation matrix"""

        return np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ]
        )

    @property
    def center(self) -> np.ndarray:
        """Access the center of the annotation as an array"""

        return np.array([self.center_x, self.center_y])

    @property
    def radii(self) -> np.ndarray:
        """Access the major and minor radii of the ellipsis"""

        return np.array([self.major_ax_radius, self.minor_ax_radius])

    @property
    def major_ax_bounds(self) -> list[float]:
        """Get the bounds coordinate on the major axis of the ellipsis
        Useful for visualisation of the major axis

        RETURNS
        -------
        bound_coordinates: list[float]
            Form [top_bound_x, top_bound_y, bot_bound_x, bot_bound_y]
        """
        major_bounds = np.tile(self.center[np.newaxis, :], (2, 1))
        major_bounds[0, 0] -= self.major_ax_radius
        major_bounds[1, 0] += self.major_ax_radius
        major_bounds = self.__rotate(major_bounds)
        return major_bounds.flatten().tolist()

    @property
    def minor_ax_bounds(self):
        """Get the bounds coordinate on the minor axis of the ellipsis
        Useful for visualisation of the minor axis

        RETURNS
        -------
        bound_coordinates: list[float]
            Form [left_bound_x, left_bound_y, right_bound_x, right_bound_y]
        """

        minor_bounds = np.tile(self.center[np.newaxis, :], (2, 1))
        minor_bounds[0, 1] -= self.minor_ax_radius
        minor_bounds[1, 1] += self.minor_ax_radius
        minor_bounds = self.__rotate(minor_bounds)
        return minor_bounds.flatten().tolist()

    @property
    def ellipsis_bbox(self):
        """Access the coordinates the top left and bottom right corner of the rotated
        ellipsis annotationg the face.
        Useful for visualising the ellipsis around the face.

        RETURNS
        -------
        bbox_corners: list[float]
            [top_left_x, top_left_y, bot_right_x, bot_right_y]
        """

        bbox_corners = np.tile(self.center[np.newaxis, :], (2, 1))
        bbox_corners[0, :] -= self.radii
        bbox_corners[1, :] += self.radii
        return bbox_corners.flatten().tolist()

    @property
    def bbox(self):
        """Access the upright bounding box containing annotated fac
        Useful for croping the face

        RETURNS
        -------
        bbox_corners: list[float]
            [top_left_x, top_left_y, bot_right_x, bot_right_y]
        """

        bbox_ellipsis_corners = np.tile(self.center[np.newaxis, :], (4, 1))
        bbox_ellipsis_corners[0, :] -= self.radii
        bbox_ellipsis_corners[1, :] += np.array(
            [-self.major_ax_radius, self.minor_ax_radius]
        )
        bbox_ellipsis_corners[2, :] += self.radii
        bbox_ellipsis_corners[3, :] += np.array(
            [self.major_ax_radius, -self.minor_ax_radius]
        )
        bbox_ellipsis_corners = self.__rotate(bbox_ellipsis_corners)

        top_left_corner = bbox_ellipsis_corners.min(axis=0).astype(int)
        bot_right_corner = bbox_ellipsis_corners.max(axis=0).astype(int)
        bbox_corners = np.concatenate([top_left_corner, bot_right_corner])
        bbox_corners = np.maximum(0, bbox_corners)

        return bbox_corners.flatten().tolist()

    def __rotate(self, points: np.ndarray) -> np.ndarray:
        """Rotate around the center of the ellipsis using the angle of the annotation

        PARAMETERS
        ----------
        points: np.ndarray
            shape (N, 2)

        RETURNS
        -------
        rotated_points: np.ndarray
            shape (N, 2)
        """

        rotated = (points - self.center[np.newaxis, :]) @ self.rot_mtx.transpose()
        rotated += self.center[np.newaxis, :]
        return rotated


class FaceImage:
    """Class to manipulate an image containing face annotations
    Useful to crop faces and non faces areas.

    PARAMETERS
    ----------
    file_path: Path
        Path to the image file
    face_annotations: list[FaceAnnotation]
        List of parameters to all the annotated faces in the image
    other_selection_params: OtherSelectionParams
        Parameters used to define how to extract non face regions
    rng: np.random.Generator | None [default, None]
        Optional, a random number generator to use for random selections
    seed: int | None [default, None]
        Optional, a seed to define an rng if one is not provided
    """

    def __init__(  # pylint: disable=R0913, R0917
        self,
        file_path: Path,
        face_annotations: list[FaceAnnotation],
        other_selection_params: OtherSelectionParams = OtherSelectionParams(),
        rng: np.random.Generator | None = None,
        seed: int | None = None,
    ):
        self.file_path = file_path
        self.size = self.read_image().size

        self.face_annotations = face_annotations
        self.faces = self.__get_faces_areas()

        self.__rng = rng if rng is not None else np.random.default_rng(seed)

        self.other_selection_params = other_selection_params
        self.others = self.__get_others_areas()

    @property
    def num_faces(self) -> int:
        """Return the number of annotated faces in FaceImage"""

        return len(self.face_annotations)

    @property
    def num_others(self) -> int:
        """Return the number of non faces that can be extracted using
        other_selection_params and the given RNG"""

        return len(self.others)

    @property
    def name(self) -> str:
        """Name of the face image, striped of the root file directory"""

        return "/".join(str(self.file_path).split("/")[-5:]).removesuffix(".jpg")

    def __get_faces_areas(self) -> list[Rectangle]:
        """Get all the annotated faces bounding box"""

        img_w, img_h = self.size

        faces_areas = []
        for face_annotation in self.face_annotations:
            (top_left_x, top_left_y, bot_right_x, bot_right_y) = face_annotation.bbox

            bot_right_x = min(bot_right_x, img_w)
            bot_right_y = min(bot_right_y, img_h)

            face_w = bot_right_x - top_left_x
            face_h = bot_right_y - top_left_y

            cropped_dim = min(img_w, img_h, max(face_w, face_h))

            num_extra_cols = min(img_w - face_w, cropped_dim - face_w)
            num_extra_rows = min(img_h - face_h, cropped_dim - face_h)

            top_left_x = top_left_x - num_extra_cols // 2
            bot_right_x = bot_right_x + num_extra_cols // 2
            if top_left_x < 0:
                bot_right_x -= top_left_x
                top_left_x = 0
            elif bot_right_x > img_w - 1:
                top_left_x -= bot_right_x - img_w + 1
                bot_right_x = img_w - 1

            top_left_y = top_left_y - num_extra_rows // 2
            bot_right_y = bot_right_y + num_extra_rows // 2
            if top_left_y < 0:
                bot_right_y -= top_left_y
                top_left_y = 0
            elif bot_right_y > img_h - 1:
                top_left_y -= bot_right_y - img_h + 1
                bot_right_y = img_h - 1

            faces_areas.append(
                Rectangle(
                    top_left_x=top_left_x,
                    top_left_y=top_left_y,
                    bot_right_x=bot_right_x,
                    bot_right_y=bot_right_y,
                )
            )

        return faces_areas

    def __get_others_areas(self) -> list[Rectangle]:
        """Get all the non face areas bounding box obtained using the
        other_selection_params and the provided rng."""

        img_w, img_h = self.size

        excluded_areas = [Rectangle(*face.bbox) for face in self.face_annotations]
        for face in excluded_areas:
            face.bot_right_x = min(face.bot_right_x, img_w)
            face.bot_right_y = min(face.bot_right_y, img_h)

        widest_face = max(map(lambda x: x.width, excluded_areas))
        highest_face = max(map(lambda x: x.height, excluded_areas))

        default_other_size = min(img_w, img_h, highest_face, widest_face)
        min_allowed_other_size = (
            self.other_selection_params.min_proportion * default_other_size
        )
        other_size = default_other_size

        found_instances = []
        while True:
            for _ in range(self.other_selection_params.max_trials):
                start_x, start_y = self.__rng.integers(
                    [0, 0], [img_w - other_size + 1, img_h - other_size + 1]
                )
                proposal = Rectangle(
                    top_left_x=start_x,
                    top_left_y=start_y,
                    bot_right_x=start_x + other_size,
                    bot_right_y=start_y + other_size,
                )

                overlap = max(
                    map(proposal.overlap_ratio, excluded_areas + found_instances)
                )

                if overlap < self.other_selection_params.max_overlap:
                    found_instances.append(proposal)

                if (
                    self.other_selection_params.num_instances is not None
                    and len(found_instances)
                    == self.other_selection_params.num_instances
                ):
                    return found_instances

            other_size = int(0.9 * other_size)
            if other_size < min_allowed_other_size:
                return found_instances

    def read_image(self) -> Image.Image:
        """Open the image"""

        return Image.open(self.file_path)

    def get_tallest_face(self) -> int:
        """Find the tallest face annotated face and return its index"""

        tallest_face = 0
        tallest_face_id = None

        for face_id, face in self.face_annotations.items():
            (_, top_left_y, _, bot_right_y) = face.bbox
            face_h = bot_right_y - top_left_y

            if face_h > tallest_face:
                tallest_face = face_h
                tallest_face_id = face_id

        return tallest_face_id

    def crop_face_img(self, face_id: int) -> Image.Image:
        """Crop the image around the bounding box of the face at index face_id"""

        return self.read_image().crop(astuple(self.faces[face_id]))

    def crop_all_faces_img(self) -> list[Image.Image]:
        """Crop all the annotated faces"""

        return [self.crop_face_img(i) for i in range(self.num_faces)]

    def crop_other_img(self, other_id) -> list[Image.Image]:
        """Crop a non face image at index other_id"""

        return self.read_image().crop(astuple(self.others[other_id]))

    def crop_all_others_img(self) -> list[Image.Image]:
        """Crop all the non faces images"""

        return [self.crop_other_img(i) for i in range(self.num_others)]

    def show_annotated_image(self) -> Image.Image:
        """Returns a PIL image with all annotations drawn on it"""

        face_img = self.read_image().convert("RGBA")
        drawing = ImageDraw.Draw(face_img)

        for face in self.face_annotations:
            drawing.line(face.major_ax_bounds, fill="blue", width=2)
            drawing.line(face.minor_ax_bounds, fill="blue", width=2)

            ellipse_img = Image.new("RGBA", face_img.size, (0, 0, 0, 0))
            ellipse_draw = ImageDraw.Draw(ellipse_img)
            ellipse_draw.ellipse(face.ellipsis_bbox, outline="blue", width=2)
            ellipse_img = ellipse_img.rotate(
                np.rad2deg(-face.angle), center=face.center.tolist()
            )
            face_img.paste(ellipse_img, (0, 0), ellipse_img)

        return face_img

    def __str__(self):
        face_word = "faces" if self.num_faces > 1 else "face"
        return f"{str(self.file_path)} has {self.num_faces} {face_word}."
