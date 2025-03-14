"""Run face detection experiments and store results"""

import os
from pathlib import Path
import sys
import time
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parents[2]))
from cv_prince.chap_07_complex_densities.gmm_face_detection import GMMFaceDetector
from experiments.datasets.faces.dataset import FaceDataset, FaceLabel


def main():
    root_data_dir = Path(os.environ.get("DATASETS"))
    root_res_dir = Path(os.environ.get("EXPERIMENTS"))

    data_dir = Path(root_data_dir) / "FDDB"
    faces_dir = data_dir / "originalPics"
    annotations_dir = data_dir / "FDDB-folds"
    res_dir = root_res_dir / "cv_prince/chap_07_complex_densities/face_detection"

    face_dataset = FaceDataset(annotations_dir=annotations_dir, imgs_dir=faces_dir)
    num_folds = face_dataset.num_folds

    fold_ids = np.arange(1, num_folds + 1)
    outer_folds = KFold(n_splits=num_folds).split(fold_ids)

    for train_val_folds_idx, test_folds_idx in outer_folds:
        train_val_folds = fold_ids[train_val_folds_idx]
        test_folds = fold_ids[test_folds_idx]

        inner_folds = KFold(n_splits=5).split(train_val_folds)
        test_fold_id = "_".join([str(fold) for fold in test_folds])

        for train_folds_idx, val_folds_idx in inner_folds:
            train_folds = train_val_folds[train_folds_idx]
            val_folds = train_val_folds[val_folds_idx]

            val_fold_id = "_".join([str(fold) for fold in val_folds])
            train_data, train_labels = face_dataset.get_data_from_folds(train_folds)

            face_detector_file = (
                res_dir
                / f"face_detector_valfolds_{val_fold_id}_testfolds_{test_fold_id}.pkl"
            )
            start_time = time.time()
            face_detector = GMMFaceDetector()
            face_detector.fit(train_data, train_labels)
            runtime = time.time() - start_time
            # face_detector.save(face_detector_file)

            val_data, val_labels = face_dataset.get_data_from_folds(val_folds)
            val_predicts = face_detector.predict(val_data)

            confusion = val_labels == val_predicts
            pos_loc = val_labels == FaceLabel.face
            num_pos = pos_loc.sum()

            neg_loc = val_labels == FaceLabel.other
            num_neg = neg_loc.sum()

            accuracy = confusion.sum() / len(val_data)
            face_tpr = confusion[pos_loc].sum() / num_pos
            face_tnr = confusion[neg_loc].sum() / num_neg
            face_fpr = np.logical_not(confusion[neg_loc]).sum() / num_neg
            face_fnr = np.logical_not(confusion[pos_loc]).sum() / num_pos

            print("MINE")
            print(runtime)
            print(f"Accuracy = {accuracy:.2%}")
            print(f"Face TPR = {face_tpr:.2%}")
            print(f"Face TNR = {face_tnr:.2%}")
            print(f"Face FPR = {face_fpr:.2%}")
            print(f"Face FNR = {face_fnr:.2%}")


if __name__ == "__main__":
    main()
