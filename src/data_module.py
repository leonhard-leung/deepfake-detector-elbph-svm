"""
data_module.py

This module defines the `DeepfakeDataset` for accessing the dataset splits for
model development and evaluation

Outputs:
1. X and y training dataset
2. X and y validation dataset
3. X and y testing dataset
"""

from preprocessing import load_numpy_images
from sklearn.model_selection import train_test_split
import numpy as np

class DeepfakeDataset:
    def __init__(self, real_images_path, fake_images_path, evaluation_split, validation_split):
        self.real_images_path = real_images_path
        self.fake_images_path = fake_images_path
        self.evaluation_split = evaluation_split
        self.validation_split = validation_split
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_eval = None
        self.y_eval = None

    def setup(self):
        real_images = load_numpy_images(self.real_images_path)
        fake_images = load_numpy_images(self.fake_images_path)

        X = np.array(real_images + fake_images)
        y_real = np.zeros(len(real_images))
        y_fake = np.zeros(len(fake_images))
        y = np.concatenate([y_real, y_fake])

        X_train_val, X_eval, y_train_val, y_eval = train_test_split(
            X, y, test_size=self.evaluation_split, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.validation_split, random_state=42, stratify=y_train_val
        )

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.X_eval, self.y_eval = X_eval, y_eval