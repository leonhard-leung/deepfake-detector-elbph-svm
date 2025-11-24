"""
data_module.py

This module defines the `DeepfakeDataset` for accessing the dataset splits for
model development and evaluation

Outputs:
1. training dataset with landmarks
2. validation dataset with landmarks
3. testing dataset with landmarks

Usage:
    from data_module import DeepfakeDataset

    dataset = DeepfakeDataset(
        real_images_path='data/processed/real',
        fake_images_path='data/processed/fake',
        real_landmarks_path='data/landmarks/real',
        fake_landmarks_path='data/landmarks/fake',
        evaluation_split=0.1,
        validation_split=0.2
    )
    dataset.setup()

    X_train, y_train, landmarks_train = dataset.get_split('train')
    X_val, y_val, landmarks_val = dataset.get_split('val')
    X_eval, y_eval, landmarks_eval = dataset.get_split('eval')
"""

from preprocessing import load_numpy_images
from sklearn.model_selection import train_test_split
import numpy as np

class DeepfakeDataset:
    def __init__(
            self,
            real_images_path,
            fake_images_path,
            real_landmarks_path,
            fake_landmarks_path,
            evaluation_split,
            validation_split
    ):
        self.real_images_path = real_images_path
        self.fake_images_path = fake_images_path
        self.real_landmarks_path = real_landmarks_path
        self.fake_landmarks_path = fake_landmarks_path
        self.evaluation_split = evaluation_split
        self.validation_split = validation_split
        self.train_dataset = None
        self.val_dataset = None
        self.eval_dataset = None

    def setup(self):
        real_images = load_numpy_images(self.real_images_path)
        fake_images = load_numpy_images(self.fake_images_path)

        real_landmarks = load_numpy_images(self.real_landmarks_path)
        fake_landmarks = load_numpy_images(self.fake_landmarks_path)

        X = np.array(real_images + fake_images)
        landmarks = np.array(real_landmarks + fake_landmarks)
        y = np.concatenate([np.zeros(len(real_images)), np.ones(len(fake_images))])

        X_train_val, X_eval, y_train_val, y_eval, landmarks_train_val, landmarks_eval = train_test_split(
            X, y, landmarks,
            test_size=self.evaluation_split,
            random_state=42,
            stratify=y
        )

        X_train, X_val, y_train, y_val, landmarks_train, landmarks_val = train_test_split(
            X_train_val, y_train_val, landmarks_train_val,
            test_size=self.validation_split,
            random_state=42,
            stratify=y_train_val
        )

        self.train_dataset = {
            'X': X_train,
            'y': y_train,
            'landmarks': landmarks_train
        }

        self.val_dataset = {
            'X': X_val,
            'y': y_val,
            'landmarks': landmarks_val
        }

        self.eval_dataset = {
            'X': X_eval,
            'y': y_eval,
            'landmarks': landmarks_eval
        }

    def get_split(self, split='train'):
        dataset_map = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'eval': self.eval_dataset
        }
        d = dataset_map[split]
        return d['X'], d['y'], d['landmarks']