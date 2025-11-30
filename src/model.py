"""
model.py

This script implements the model development pipeline for deepfake detection using
enhanced Local Binary Pattern Histograms (eLBPH) features and a Support Vector
Machine (SVM) classifier.

Features:
1. Loads preprocessed images and cached facial landmarks from disk.
2. Splits the dataset into training, validation, and evaluation sets.
3. Extracts eLBPH features with multi-scale LBP and spatial weighting based on facial landmarks.
4. Applies PCA for dimensionality reduction.
5. Trains an SVM classifier with RBF kernel.
6. Evaluates the model on training, validation and evaluation sets, displaying metrics and confusion matrices.
7. Saves the trained model to disk for future inference.

Usage:
    - `python src/model.py` : terminal
"""

from data_module import DeepfakeDataset
from elbph import extract_lbp_features
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
import os

def _train(model, X, y):
    """
    Train an SVM on LBP features with PCA dimensionality reduction.

    :param model: SVM model from sklearn.svm.SVC
    :param X: 2D numpy array of training features (samples x features)
    :param y: 1D numpy array of labels (0 = Real, 1 = Fake)
    :return: None
    """
    model.fit(X, y)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\nTraining Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def _validate(model, X, y):
    """
    Evaluate the SVM model on a validation set.

    :param model: Trained SVM model
    :param X: 2D numpy array of validation features
    :param y: 1D numpy array of validation labels
    :return: None
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\nValidation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def _evaluate(model, X, y):
    """
    Evaluate the SVM model on the unseen/evaluation data.

    :param model: Trained SVM model
    :param X: 2D numpy array of evaluation features
    :param y: 1D numpy array of test labels
    :return: None
    """
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\nEvaluation Metrics:")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")

    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Real', 'Fake'])
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def _save_model(model, output_dir='../model'):
    """
    Save the trained SVM model.

    :param model: Trained SVM model
    :param output_dir: Directory to save the model
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)
    filepath = output_dir / 'svm_deepfake_model.joblib'
    joblib.dump(model, filepath)
    print(f"Model saved as {filepath}")

def main():
    """
    Full pipeline for training and evaluating a deepfake detection SVM classifier using eLBPH features.

    :return: None
    """
    # instantiate dataset
    dataset = DeepfakeDataset(
        real_images_path="../data/processed/real",
        fake_images_path="../data/processed/fake",
        real_landmarks_path="../data/landmarks/real",
        fake_landmarks_path="../data/landmarks/fake",
        evaluation_split=0.1,
        validation_split=0.2,
    )
    dataset.setup()

    # obtain the train and validation sets
    X_train, y_train, landmarks_train = dataset.get_split('train')
    X_val, y_val, landmarks_val = dataset.get_split('val')
    X_eval, y_eval, landmarks_eval = dataset.get_split('eval')

    # extract features using eLBPH
    print("Extracting eLBPH features for training set...")
    X_train_features = extract_lbp_features(
        images=X_train,
        landmarks_list=landmarks_train,
        scales=[(8,1), (16,2), (24,3)],
        num_blocks=(8,8)
    )

    print("Extracting eLBPH features for validation set...")
    X_val_features = extract_lbp_features(
        images=X_val,
        landmarks_list=landmarks_val,
        scales=[(8, 1), (16, 2), (24, 3)],
        num_blocks=(8, 8)
    )

    print("Extracting eLBPH features for evaluation set...")
    X_eval_features = extract_lbp_features(
        images=X_eval,
        landmarks_list=landmarks_eval,
        scales=[(8, 1), (16, 2), (24, 3)],
        num_blocks=(8, 8)
    )

    # dimensionality reduction
    n_components = 200
    pca = PCA(n_components=n_components)
    X_train_features = pca.fit_transform(X_train_features)
    X_val_features = pca.transform(X_val_features)
    X_eval_features = pca.transform(X_eval_features)

    # save PCA model for inference
    os.makedirs("../model", exist_ok=True)
    joblib.dump(pca, "../model/pca_model.joblib")
    print("PCA model saved to '../model/pca_model.joblib'")

    # instantiate SVM model
    model = SVC(kernel='rbf', probability=True)

    # train model
    _train(model, X_train_features, y_train)

    # validate model
    _validate(model, X_val_features, y_val)

    # evaluate model
    _evaluate(model, X_eval_features, y_eval)

    # save model
    _save_model(model, output_dir='../model')

if __name__ == "__main__":
    main()