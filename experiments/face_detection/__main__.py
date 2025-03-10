"""Run face detection experiments and store results"""

import os
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from cv_prince.chap_07_complex_densities.gmm_face_detection import GMMFaceDetector
from experiments.datasets.faces.dataset import FaceDataset, FaceLabel


def main():
    root_data_dir = Path(os.environ.get("DATASETS"))
    root_res_dir = Path(os.environ.get("EXPERIMENTS"))

    data_dir = Path(root_data_dir) / "FDDB"
    faces_dir = data_dir / "originalPics"
    annotations_dir = data_dir / "FDDB-folds"

    face_dataset = FaceDataset(annotations_dir=annotations_dir, imgs_dir=faces_dir)

    # test_fold = [10]
    val_folds = [8, 9]
    train_folds = [1, 2, 3, 4, 5, 6, 7]

    face_dataset = FaceDataset(annotations_dir=annotations_dir, imgs_dir=faces_dir)
    train_data, train_labels = face_dataset.get_data_from_folds(train_folds)

    res_dir = root_res_dir / "cv_prince/chap_07_complex_densities/face_detection"
    face_detector_file = res_dir / "face_detector.pkl"
    face_detector = GMMFaceDetector()

    face_detector.fit(train_data, train_labels)
    face_detector.save(face_detector_file)

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

    print(f"Accuracy = {accuracy:.2%}")
    print(f"Face TPR = {face_tpr:.2%}")
    print(f"Face TNR = {face_tnr:.2%}")
    print(f"Face FPR = {face_fpr:.2%}")
    print(f"Face FNR = {face_fnr:.2%}")


if __name__ == "__main__":
    main()
