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


def find_corners(points):
    tl = np.array([points[:, 0].min(), points[:, 1].min()])
    br = np.array([points[:, 0].max(), points[:, 1].max()])

    return tl.astype(int), br.astype(int)


def offset_mean_face(mean_landmarks, offset_percentage=[0.7, 0.7]):
    tl_corner, br_corner = find_corners(mean_landmarks)

    width = (br_corner - tl_corner)[0]
    height = (br_corner - tl_corner)[1]

    offset = np.array([int(offset_percentage[0] * width), int(offset_percentage[1] * height)])
    return mean_landmarks - tl_corner + offset


def find_mean_face(fa, directory, offset_percentage=[0.7, 0.7]):
    mean_landmarks = np.zeros([68, 2])
    number_of_faces = 0
    files = Path(directory).glob('*.jpg')
    for i, frame in enumerate(tqdm(files, desc="finding mean face ...")):
        if i % 100:
            number_of_faces += 1
            mean_landmarks += fa.get_landmarks(str(frame))[0]

        if number_of_faces == 1000:
            break

    mean_face = np.multiply(1 / number_of_faces, mean_landmarks)

    mean_face = offset_mean_face(mean_face, offset_percentage)

    return mean_face


def warp_img(src, dst, img, output_shape=None):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=output_shape)  # wrap the frame image
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def align_directory(directory, outdir, device='cuda'):
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
        except:
            return None

        stable_points = landmarks[stablePntsIDs, :]

        warped_img, transf = warp_img(stable_points, mean_face[stablePntsIDs, :], cv2.imread(frame_name))
        # saves warped_img & new landmarks
        cv2.imwrite(os.path.join(outdir, frame_name_number + '.jpg'), warped_img)
