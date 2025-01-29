"""Utility functions for notebooks"""

import numpy as np


def rotation_mtx_from_angle(theta: float) -> np.ndarray:
    """Create a 2d rotation matrix from a given angle"""

    theta_rad = np.deg2rad(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    return np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])


def create_cov_based_on_angle_and_axis_scale(
    angle: float, scale: tuple[float, float]
) -> np.ndarray:
    """Create a covariance matrix with specific orientation and scale"""

    rot_mtx = rotation_mtx_from_angle(angle)
    scale_mtx = np.diag(scale)

    return rot_mtx @ scale_mtx @ np.linalg.inv(rot_mtx)
