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

Notes:
    - This script uses functions from preprocessing.py to preprocess images.
    - Do not run during training; train.py uses the processed images which this script outputs.
"""

from preprocessing import load_images, clean_images, save_as_numpy_images

def main():
    raw_images_fake = load_images('../data/raw/fake')
    raw_images_real = load_images('../data/raw/real')

    processed_images_fake = clean_images(raw_images_fake)
    processed_images_real = clean_images(raw_images_real)

    save_as_numpy_images(processed_images_fake, '../data/processed/fake')
    save_as_numpy_images(processed_images_real, '../data/processed/real')
    print("Images processed and saved...")

if __name__ == "__main__":
    main()
