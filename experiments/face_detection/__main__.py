"""Run face detection experiments and store results"""

import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# pylint: disable=wrong-import-position
sys.path.insert(0, str(Path(__file__).parents[2]))
from cv_prince.chap_07_complex_densities.gmm_face_detection import (  # noqa E402
    GMMFaceDetector,
    GMMFaceDetectorParams,
)
from experiments.datasets.faces.dataset import FaceDataset, FaceLabel  # noqa E402


def main():  # pylint: disable=R0914, C0116
    root_data_dir = Path(os.environ.get("DATASETS"))

    data_dir = Path(root_data_dir) / "FDDB"
    faces_dir = data_dir / "originalPics"
    annotations_dir = data_dir / "FDDB-folds"

    face_dataset = FaceDataset(annotations_dir=annotations_dir, imgs_dir=faces_dir)
    num_folds = face_dataset.num_folds

    fold_ids = np.arange(1, num_folds + 1)
    outer_folds = KFold(n_splits=5).split(fold_ids)

    num_components = [3, 5, 10]

    for train_val_folds_idx, test_folds_idx in outer_folds:
        train_val_folds = fold_ids[train_val_folds_idx]
        test_folds = fold_ids[test_folds_idx]

        inner_folds = KFold(n_splits=5).split(train_val_folds)

        inner_aurocs = defaultdict(list)

        for train_folds_idx, val_folds_idx in inner_folds:
            train_folds = train_val_folds[train_folds_idx]
            val_folds = train_val_folds[val_folds_idx]

            train_data, train_labels = face_dataset.get_data_from_folds(train_folds)
            for num_component in num_components:
                face_detector = GMMFaceDetector(
                    GMMFaceDetectorParams(
                        ncomponents_faces=num_component,
                        ncomponents_others=num_component,
                    )
                )
                face_detector.fit(train_data, train_labels)

                val_data, val_labels = face_dataset.get_data_from_folds(val_folds)
                val_scores = face_detector.score(val_data)

                inner_aurocs[num_component].append(
                    roc_auc_score(val_labels == FaceLabel.FACE, val_scores)
                )

        mean_inner_aurocs = list(map(np.mean, inner_aurocs.values()))
        best_model_id = np.argmax(mean_inner_aurocs)
        best_num_components = num_components[best_model_id]
        best_mean_auroc = mean_inner_aurocs[best_model_id]
        best_std_auroc = np.std(inner_aurocs[best_num_components])
        assert best_mean_auroc == np.mean(inner_aurocs[best_num_components])

        train_data, train_labels = face_dataset.get_data_from_folds(train_val_folds)
        face_detector = GMMFaceDetector(
            GMMFaceDetectorParams(
                ncomponents_faces=best_num_components,
                ncomponents_others=best_num_components,
            )
        )
        face_detector.fit(train_data, train_labels)
        test_data, test_labels = face_dataset.get_data_from_folds(test_folds)
        test_scores = face_detector.score(test_data)

        test_auroc = roc_auc_score(test_labels == FaceLabel.FACE, test_scores)

        print(f"Best model (num components)  : {best_num_components}")
        print(f"Best model (inner mean AUROC): {best_mean_auroc:.2%}")
        print(f"Best model (inner std AUROC) : {best_std_auroc:.2%}")
        print(f"Test AUROC (outer split)     : {test_auroc:.2%}")
        print("==========================================================\n")


if __name__ == "__main__":
    main()
