"""
prepare_data.py

This script performs the preprocessing pipeline on the raw images.
It should be run **once** before training the SVM model to generate
the preprocessed images in numpy array format.

Usage:
    - `python src/prepare_data.py` : terminal

Outputs:
    - `data/processed/fake` : preprocessed fake images
    - `data/processed/real` : preprocessed real images
    - `data/landmarks/fake` : preprocessed landmarks images
    - `data/landmarks/real` : preprocessed real landmarks images

Notes:
    - This script uses functions from preprocessing.py to preprocess images.
    - Do not run during training; train.py uses the processed images which this script outputs.
"""

from preprocessing import load_images, preprocess_images, save_as_numpy_images, save_landmark_dicts

def main():
    raw_images_fake = load_images(r'D:\archive\images\fake')
    raw_images_real = load_images(r'D:\archive\images\real')

    print("Applying preprocessing for fake images...")
    processed_images_fake, processed_landmarks_fake = preprocess_images(raw_images_fake)
    print("Applying preprocessing for real images...")
    processed_images_real, processed_landmarks_real = preprocess_images(raw_images_real)

    save_as_numpy_images(processed_images_fake, '../data/processed/fake')
    save_as_numpy_images(processed_images_real, '../data/processed/real')

    save_landmark_dicts(processed_landmarks_fake, '../data/landmarks/fake')
    save_landmark_dicts(processed_landmarks_real, '../data/landmarks/real')

    print("Images and landmarks processed and saved...")

if __name__ == "__main__":
    main()
