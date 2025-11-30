"""
preprocessing.py

This module provides reusable functions for image preprocessing, including:
- Face detection and alignment using facenet-pytorch MTCNN (GPU-supported)
- Image resizing to a fixed shape
- Conversion from BGR to Y (luminance) channel from YCrCb color space
- Gaussian filtering to reduce noise and enhance textures.
- Batch Preprocessing pipelines that return preprocessed images along with face landmarks
- File handling for loading and saving images and landmarks as NumPy arrays

Parallel processing is used for faster execution on large datasets.

Intended to be **imported** by scripts such as `prepare_daya.py` for preprocessing
datasets before feature extraction and ML modeling.

Usage:
    import preprocessing

    images = preprocessing.load_images('../data/raw')
    processed_images = preprocessing.preprocess_images(images)
"""

import torch
import numpy as np
import cv2 as cv
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# =============== setup GPU MTCNN ===============
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn_detector = MTCNN(keep_all=False, device=device) # global instantiation of mtcnn to avoid multiple instantiations

# =============== preprocessing techniques ===============
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

def _crop_face(img, padding=0.4):
    """
    Aligns the face in an image using MTCNN by rotating the eyes to be horizontal
    and cropping the face with optional padding.

    :param img: Input image in BGR format
    :param padding: Fraction of face width/height to add as padding around the crop, default is 0.4
    :return: Cropped and aligned face image
    """
    img_pil = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    boxes, probs, landmarks = mtcnn_detector.detect(img_pil, landmarks=True)

    if boxes is None:
        return img, {}

    box = boxes[0].astype(int)
    x1, y1, x2, y2 = box

    w, h = x2 - x1, y2 - y1
    x1 = max(int(x1 - w * padding), 0)
    y1 = max(int(y1 - h * padding), 0)
    x2 = min(int(x2 + w * padding), img.shape[1])
    y2 = min(int(y2 + h * padding), img.shape[0])

    cropped = img[y1:y2, x1:x2]

    landmark_dict = {}
    if landmarks is not None:
        landmark_dict = {
            'left_eye': tuple(landmarks[0][0]),
            'right_eye': tuple(landmarks[0][1]),
            'nose': tuple(landmarks[0][2]),
            'mouth_left': tuple(landmarks[0][3]),
            'mouth_right': tuple(landmarks[0][4])
        }

    return cropped, landmark_dict

def _gaussian_filter(img):
    """
    Applies mild Gaussian blur to reduce noise and enhance textures.

    :param img: Input image in BGR format
    :return: Filtered image
    """
    return cv.GaussianBlur(img, (3,3), 1)

def _to_luminance_channel(img):
    """
    Convert an image from BGR to luminance channel from YCrCb.

    :param img: Input image in BGR format
    :return: Image in luminance (Y) color space
    """
    y_cr_cb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y = y_cr_cb[:, :, 0]

    return y

# =============== preprocessing pipeline ===============
def preprocess_images(images):
    """
    Apply a full preprocessing pipeline to a list of images using facenet-pytorch MTCNN
    1. Face detection
    2. Resizing to a fixed shape (default: 256x256)
    3. Conversion from BGR to Y channel (luminance) of YCrCb color space.
    4. Mild Gaussian filtering to reduce noise and enhance textures.
    5. Parallel processing using ThreadPoolExecutor for faster execution on large datasets.

    :param images: List of input images in BGR format
    :return:
        - cleaned: List of preprocessed images in Y (luminance) channel, resized, cropped, and filtered
        - landmarks_list: List of dictionaries containing facial keypoints for each image; empty dict if no face detected
    """
    cleaned, landmarks_list = [], []

    def process(img):
        cropped, landmarks_dict = _crop_face(img, padding=0.4)
        resized = _resize(cropped)
        y_channel = _to_luminance_channel(resized)
        filtered = _gaussian_filter(y_channel)
        return filtered, landmarks_dict

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process, images), total=len(images), desc="Preprocessing images"))

    cleaned, landmarks_list = zip(*results)
    return list(cleaned), list(landmarks_list)

# =============== file handling ===============
def load_images(input_dir='../data/raw'):
    """
    Load all image files (jpg, png, jpeg) from a directory.

    :param input_dir: Directory containing image files, default is '../data/raw'
    :return: List of loaded images in BGR format
    """
    input_path = Path(input_dir)
    images = []

    files = []
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        files.extend(list(input_path.glob(ext)))

    for file in tqdm(files, desc=f"Loading images from {input_dir}"):
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
    for idx, img in enumerate(tqdm(images, desc=f"Saving images to {output_dir}")):
        filename = output_path / f'img_{idx}.npy'
        np.save(str(filename), img)

def load_numpy_images(input_dir='../data/processed'):
    """
    Load preprocessed images saved as .npy arrays from a directory.

    :param input_dir: Directory containing .npy files, default is '../data/processed'
    :return: List of loaded images in numpy arrays
    """
    input_path = Path(input_dir)
    files = list(input_path.glob("*.npy"))
    arrays = []

    for file in tqdm(files, desc=f"Loading numpy images from {input_dir}"):
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

    for idx, lm in enumerate(tqdm(landmarks_list, desc=f"Saving landmarks to {output_dir}")):
        np.save(output_path / f'landmarks_{idx}.npy', lm)

def load_landmarks(input_dir='../data/processed'):
    """
    Load precomputed face landmarks saved as .npy files.

    :param input_dir: Directory containing .npy files, default is '../data/processed'
    :return: List of dictionaries, one per image, containing facial keypoints or empty dictionary if no face is detected
    """
    input_path = Path(input_dir)
    files = sorted(input_path.glob("*.npy"))
    landmarks_list = []

    for file in tqdm(files, desc=f"Loading landmarks from {input_dir}"):
        landmarks_list.append(np.load(file, allow_pickle=True).item())
    return landmarks_list