import argparse
import json
import os
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from skimage import transform as tf
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

stablePntsIDs = [33, 36, 39, 42, 45]


def find_corners(points: np.ndarray) -> tuple[int, int]:
    """
    Finds corner points from a set of face points
    :param points: np.ndarray [shape=(n,m)]
            Array of face points
    :return: tuple(int,int)
            top left corner, bottom right corner
    """
    tl = np.array([points[:, 0].min(), points[:, 1].min()])
    br = np.array([points[:, 0].max(), points[:, 1].max()])

    return tl.astype(int), br.astype(int)


def offset_mean_face(mean_landmarks: np.ndarray,
                     offset_percentage: list[float, float] = [0.7, 0.7]) -> np.ndarray:
    """
    Adds an offset (offset_percentage) to the mean face landmarks.
    :param mean_landmarks: np.ndarray[(n,m)]
            Mean landmarks of the whole face image set.
    :param offset_percentage: list[float, float]
            Offset percentage [x-coordinate, y-coordinate]
    :return: np.ndarray[(n,m)]
            Mean face landmarks with offset.
    """
    tl_corner, br_corner = find_corners(mean_landmarks)

    width = (br_corner - tl_corner)[0]
    height = (br_corner - tl_corner)[1]

    offset = np.array([int(offset_percentage[0] * width), int(offset_percentage[1] * height)])
    return mean_landmarks - tl_corner + offset


def find_mean_face(fa, directory: str, offset_percentage: list[float, float] = [0.7, 0.7], step: int = 100):
    """
    Calculates the mean face from in directory with a default step of 100. The maximum number of image to take into
    account is 1000.
    :param step: int
            Calculates mean face for every step-th image in the directory.
    :param fa: <class 'FaceAlignment'>
            FaceAlignment class object initialized as face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
            flip_input=False, device=device)
    :param directory: str
            Directory containing face images.
    :param offset_percentage: list[float, float]
            Offset percentage [x-coordinate, y-coordinate]
    :return:
    """
    if step == 0:
        step = 1

    mean_landmarks = np.zeros([68, 2])
    number_of_faces = 0
    files = Path(directory).glob('*.jpg')
    for i, frame in enumerate(tqdm(files, desc="finding mean face ...")):
        if i % step == 0:
            number_of_faces += 1
            mean_landmarks += fa.get_landmarks(str(frame))[0]

        if number_of_faces == 1000:
            break

    mean_face = np.multiply(1 / number_of_faces, mean_landmarks)

    mean_face = offset_mean_face(mean_face, offset_percentage)

    return mean_face


def warp_img(src: np.ndarray, dst: np.ndarray, img, output_shape=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Wraps image using skimage transform similarity to map from src face landmarks to dst face landmarks.
    :param src: np.ndarray[(n,m)]
            Original face landmarks positions.
    :param dst: np.ndarray[(n,m)]
            Final face landmarks positions.
    :param img: np.ndarray
            Target image.
    :param output_shape:
    :return: tuple(np.ndarray, np.ndarray)
            Warped image, similarity transform matrix.
    """
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=output_shape)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def align_directory(directory: str, outdir: str, device: str = 'cuda') -> None:
    """
    Aligns all images of faces from directory, storing the results in outdir. CPU or GPU can be selected by device.
    :param directory: str
            Directory with all face images.
    :param outdir: str ('cuda' or 'cpu')
            Output directory.
    :param device: str
    :return:
    """
    assert device is 'cuda' or 'cpu', "Device must be 'cpu' or 'cuda'"

    os.makedirs(outdir, exist_ok=True)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    # checks if mean_face exists
    mean_face = find_mean_face(fa, directory)

    files = Path(directory).glob('*.jpg')
    for frame in tqdm(files, desc="aligning frames ..."):
        frame_name = str(frame)
        frame_name_number = frame_name.split("/")[-1].split(".")[0]
        try:
            landmarkset = fa.get_landmarks(frame_name)
            landmarks = landmarkset[0]
        except NotImplementedError:
            print("No landmarks detected")
            return None

        stable_points = landmarks[stablePntsIDs, :]

        warped_img, transf = warp_img(stable_points, mean_face[stablePntsIDs, :], cv2.imread(frame_name))
        # saves warped_img & new landmarks
        cv2.imwrite(os.path.join(outdir, frame_name_number + '.jpg'), warped_img)
