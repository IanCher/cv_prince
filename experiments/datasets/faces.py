"""Classes and functions to manipulate face dataset"""

from dataclasses import astuple, dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.geometry.rectangles import Rectangle


@dataclass
class FaceAnnotation:
    major_ax_radius: float
    minor_ax_radius: float
    angle: float
    center_x: float
    center_y: float

    @classmethod
    def from_str(cls, datastr: str):
        anno = [float(coord) for coord in datastr.split()[:-1]]
        return cls(*anno)

    @cached_property
    def rot_mtx(self):
        return np.array(
            [
                [np.cos(self.angle), -np.sin(self.angle)],
                [np.sin(self.angle), np.cos(self.angle)],
            ]
        )

    @property
    def center(self):
        return np.array([self.center_x, self.center_y])

    @property
    def radii(self):
        return np.array([self.major_ax_radius, self.minor_ax_radius])

    @property
    def major_ax_bounds(self):
        major_bounds = np.tile(self.center[np.newaxis, :], (2, 1))
        major_bounds[0, 0] -= self.major_ax_radius
        major_bounds[1, 0] += self.major_ax_radius
        major_bounds = self.__rotate(major_bounds)
        return major_bounds.flatten().tolist()

    @property
    def minor_ax_bounds(self):
        minor_bounds = np.tile(self.center[np.newaxis, :], (2, 1))
        minor_bounds[0, 1] -= self.minor_ax_radius
        minor_bounds[1, 1] += self.minor_ax_radius
        minor_bounds = self.__rotate(minor_bounds)
        return minor_bounds.flatten().tolist()

    @property
    def ellipsis_bbox(self):
        bbox_corners = np.tile(self.center[np.newaxis, :], (2, 1))
        bbox_corners[0, :] -= self.radii
        bbox_corners[1, :] += self.radii
        return bbox_corners.flatten().tolist()

    @property
    def bbox(self):
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

    def __rotate(self, points: np.ndarray):
        """points of shape (N, 2)"""

        rotated = (points - self.center[np.newaxis, :]) @ self.rot_mtx.transpose()
        rotated += self.center[np.newaxis, :]
        return rotated


class FaceImage:
    def __init__(self, file_path: Path, num_faces: int):
        self.file_path = file_path
        self.num_faces = num_faces
        self.faces: dict[int, FaceAnnotation] = {}

    @cached_property
    def size(self):
        return self.read_image().size  # (width, height)

    def initialise_rng(self, seed: int | None = None):
        print("Initialising rng")
        self.__rng = np.random.default_rng(seed=seed)

    def add_face(self, anno: FaceAnnotation):
        face_id = len(self.faces)
        self.faces[face_id] = anno

    def read_image(self) -> Image.Image:
        return Image.open(self.file_path)

    def get_tallest_face(self) -> int:
        tallest_face = 0
        tallest_face_id = None

        for face_id, face in self.faces.items():
            (_, top_left_y, _, bot_right_y) = face.bbox
            face_h = bot_right_y - top_left_y

            if face_h > tallest_face:
                tallest_face = face_h
                tallest_face_id = face_id

        return tallest_face_id

    def crop_face_img(self, face_id: int) -> Image.Image:
        img_w, img_h = self.size

        (top_left_x, top_left_y, bot_right_x, bot_right_y) = self.faces[face_id].bbox
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

        return self.read_image().crop(
            (top_left_x, top_left_y, bot_right_x, bot_right_y)
        )

    def get_all_croped_faces(self):
        for i in self.faces.keys():
            yield self.crop_face_img(i)

    def crop_non_face_img(
        self,
        num_instances: int | None = None,
        max_trials: int = 100,
        max_overlap: float = 0.1,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ):

        img_w, img_h = self.size
        img = self.read_image()

        excluded_areas = [Rectangle(*face.bbox) for face in self.faces.values()]
        for face in excluded_areas:
            face.bot_right_x = min(face.bot_right_x, img_w)
            face.bot_right_y = min(face.bot_right_y, img_h)

        widest_face = max(map(lambda x: x.width, excluded_areas))
        highest_face = max(map(lambda x: x.height, excluded_areas))
        non_face_size = min(img_w, img_h, highest_face, widest_face)

        if rng is None:
            rng = np.random.default_rng(seed=seed)

        found_instances = []
        while True:
            for _ in range(max_trials):
                start_x, start_y = rng.integers(
                    [0, 0], [img_w - non_face_size + 1, img_h - non_face_size + 1]
                )
                proposal = Rectangle(
                    top_left_x=start_x,
                    top_left_y=start_y,
                    bot_right_x=start_x + non_face_size,
                    bot_right_y=start_y + non_face_size,
                )

                overlap = max(map(proposal.overlap_ratio, excluded_areas))

                if overlap < max_overlap:
                    excluded_areas.append(proposal)
                    found_instances.append(img.crop(astuple(proposal)))

                if num_instances is not None and len(found_instances) == num_instances:
                    return found_instances

            non_face_size = int(0.9 * non_face_size)
            if non_face_size < 0.25 * min(highest_face, widest_face):
                return found_instances

    def show_annotated_image(self):
        face_img = self.read_image().convert("RGBA")
        drawing = ImageDraw.Draw(face_img)

        for face_anno in self.faces.values():
            drawing.line(face_anno.major_ax_bounds, fill="blue", width=2)
            drawing.line(face_anno.minor_ax_bounds, fill="blue", width=2)

            ellipse_img = Image.new("RGBA", face_img.size, (0, 0, 0, 0))
            ellipse_draw = ImageDraw.Draw(ellipse_img)
            ellipse_draw.ellipse(face_anno.ellipsis_bbox, outline="blue", width=2)
            ellipse_img = ellipse_img.rotate(
                np.rad2deg(-face_anno.angle), center=face_anno.center.tolist()
            )
            face_img.paste(ellipse_img, (0, 0), ellipse_img)

        return face_img

    def __str__(self):
        face_word = "faces" if self.num_faces > 1 else "face"
        return f"{str(self.file_path)} has {self.num_faces} {face_word}."
