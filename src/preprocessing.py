"""
preprocessing.py

This module provides reusable functions for image preprocessing, including:
- Face alignment using MTCNN
- Image resizing
- Color space conversion (BGR to YCrCb)
- Image filters (Gaussian Filter)
- Batch Preprocessing pipelines that return preprocessed images along with face landmarks
- File handling for loading and saving images and landmarks as NumPy arrays

Intended to be **imported** by scripts such as `prepare_daya.py` for preprocessing
datasets before feature extraction and ML modeling.

Usage:
    import preprocessing

    images = preprocessing.load_images('../data/raw')
    processed_images = preprocessing.clean_images(images)
"""

import numpy as np
import cv2 as cv
from mtcnn import MTCNN
from pathlib import Path

# =============== preprocessing techniques ===============
mtcnn_detector = MTCNN() # global instantiation of mtcnn to avoid multiple instantiations

def _resize(img, resize_shape=(256, 256)):
    """
    Resizes an image to the specified dimensions using appropriate interpolation.

    :param img: Input image in BGR format
    :param resize_shape: Desired output shape as (height, width), default is (256, 256)
    :return: Resized image
    """
    height, width = img.shape[:2]
    h_res, w_res = resize_shape
    interpolation = cv.INTER_LINEAR

    if height > h_res or width > w_res:
        interpolation = cv.INTER_AREA
    elif height < h_res or width < w_res:
        interpolation = cv.INTER_CUBIC

    return cv.resize(img, resize_shape, interpolation=interpolation)

def _align_face(img, padding=0.4):
    """
    Aligns the face in an image using MTCNN by rotating the eyes to be horizontal
    and cropping the face with optional padding.

    :param img: Input image in BGR format
    :param padding: Fraction of face width/height to add as padding around the crop, default is 0.4
    :return: Cropped and aligned face image
    """
    result = mtcnn_detector.detect_faces(img)

    if len(result) == 0:
        return img

    face = result[0]['box']
    keypoints = result[0]['keypoints']
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    eyes_center = (
        int((left_eye[0] + right_eye[0]) / 2),
        int((left_eye[1] + right_eye[1]) / 2)
    )

    M = cv.getRotationMatrix2D(eyes_center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (img.shape[1], img.shape[0]))

    x, y, w, h = face
    padding_w = int(padding * w)
    padding_h = int(padding * h)
    x_new = max(x - padding_w, 0)
    y_new = max(y - padding_h, 0)
    w_new = max(w + 2 * padding_w, img.shape[1] - x_new)
    h_new = max(h + 2 * padding_h, img.shape[0] - y_new)

    cropped = rotated[y_new:y_new + h_new, x_new:x_new + w_new]
    return cropped

def _apply_filters(img):
    """
    Applies mild Gaussian blur to reduce noise and enhance textures.

    :param img: Input image in BGR format
    :return: Filtered image
    """
    return cv.GaussianBlur(img, (3,3), 1)

def _to_y_cr_cb(img):
    """
    Convert an image from BGR to YCrCb color space.

    :param img: Input image in BGR format
    :return: Image in YCrCb color space
    """
    return cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

# =============== preprocessing pipeline ===============
def clean_images(images):
    """
    Apply a full preprocessing pipeline to a list of images, including:
    1. Face alignment
    2. Resizing
    3. Color conversion (BGR -> YCrCb)
    4. Gaussian filtering

    Also detects and returns facial landmarks corresponding to each image after face alignment.

    :param images: List of input images in BGR format
    :return:
        - cleaned: List of preprocessed images in YCrCb color space, resized, aligned, and filtered
        - landmarks: List of dictionaries containing facial keypoints for each image
    """
    cleaned = []
    landmarks = []

    for img in images:
        img_aligned = _align_face(img)

        result = mtcnn_detector.detect_faces(img_aligned)
        if result:
            landmarks.append(result[0]['keypoints'])
        else:
            landmarks.append({})

        img_resized = _resize(img_aligned)
        img_color_space = _to_y_cr_cb(img_resized)
        img_filtered = _apply_filters(img_color_space)
        cleaned.append(img_filtered)

    return cleaned, landmarks

# =============== file handling ===============
def load_images(input_dir='../data/raw'):
    """
    Load all image files (jpg, png, jpeg) from a directory.

    :param input_dir: Directory containing image files, default is '../data/raw'
    :return: List of loaded images in BGR format
    """
    input_path = Path(input_dir)
    images = []

    for ext in ('*.jpg', '*.png', '*.jpeg'):
        for file in input_path.glob(ext):
            img = cv.imread(str(file))
            if img is not None:
                images.append(img)

    return images

def save_as_numpy_images(images, output_dir='../data/processed'):
    """
    Save a list of images as .npy files for later processing.

    :param images: List of preprocessed images
    :param output_dir: Directory to save .npy files, default is '../data/processed'
    :return: None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, img in enumerate(images):
        filename = output_path / f'img_{idx}.npy'
        np.save(str(filename), img)

def load_numpy_images(input_dir='../data/processed'):
    """
    Load preprocessed images saved as .npy arrays from a directory.

    :param input_dir: Directory containing .npy files, default is '../data/processed'
    :return: List of loaded images in numpy arrays
    """
    input_path = Path(input_dir)
    arrays = []

    for file in input_path.glob('*.npy'):
        arrays.append(np.load(str(file)))

    return arrays

def save_landmark_dicts(landmarks_list, output_dir='../data/processed'):
    """
    Save a list of facial landmark dictionaries as .npy files.

    :param landmarks_list: List of dictionaries containing facial keypoints
    :param output_dir: Directory to save landmarks, default is '../data/processed'
    :return: None
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for idx, lm in enumerate(landmarks_list):
        np.save(output_path / f'landmarks_{idx}.npy', lm)

def load_landmarks(input_dir='../data/preprocessed'):
    """
    Load precomputed face landmarks saved as .npy files.

    :param input_dir: Directory containing .npy files, default is '../data/processed'
    :return: List of dictionaries, one per image, containing facial keypoints or empty dictionary if no face is detected
    """
    input_path = Path(input_dir)
    landmarks_list = []
    for file in sorted(input_path.glob('*.npy')):
        landmarks_list.append(np.load(file, allow_pickle=True).item())
    return landmarks_list