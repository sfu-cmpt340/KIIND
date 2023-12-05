import sys
import json
import base64
import numpy as np
import os
from PIL import Image
import io
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay, auc


def main():
    data = load_data('MRNet-v1.0/train')
    labels = load_labels('MRNet-v1.0/train-abnormal.csv', 'MRNet-v1.0/train-acl.csv', 'MRNet-v1.0/train-meniscus.csv')
    # Assuming 'data' is your array of MRI scans
    batch_size = 10  # Set your batch size

    # Initialize a new array to hold the cropped images
    cropped_data = np.zeros((data.shape[0], data.shape[1], 160, 160, data.shape[-1]), dtype=data.dtype)

    # Loop over the data in batches for cropping
    for start in range(0, len(data), batch_size):
        end = start + batch_size
        batch = data[start:end]

        # Apply cropping to each image in the batch
        cropped_batch = np.array([crop_center(img, 160, 160) for img in batch])

        # Update the cropped_data array with the cropped_batch
        cropped_data[start:end] = cropped_batch

        # Feedback to user
        print(f"Cropped batch from index {start} to {end}")

    data = cropped_data
    model_path = os.path.join('../models/imageclassifier11.h5')
    model = load_model(model_path)

    y_pred = model.predict(data)
    display = []
    optimal_thresholds = find_optimal_thresholds(labels, y_pred)
    # plot ROC curve
    display[0].plot()
    display[1].plot()
    display[2].plot()

    y_pred, correctness = apply_thresholds_and_evaluate(labels, y_pred, optimal_thresholds)



def apply_thresholds_and_evaluate(y_true, y_pred_probs, thresholds):
    # Apply thresholds to prediction probabilities to create binary predictions
    y_pred = (y_pred_probs >= thresholds).astype(int)
    
    # Determine which predictions match the true labels
    correctness = (y_pred == y_true).astype(int)
    
    # Calculate overall correctness (sum of correct predictions for each label)
    overall_correctness = np.sum(correctness)
    
    # Calculate accuracy for each label
    accuracy_per_label = np.mean(correctness, axis=0)
    
    # Calculate overall accuracy (proportion of correct predictions across all labels)
    overall_accuracy = overall_correctness / np.prod(y_true.shape)
    
    print(f"Overall Correctness: {overall_correctness} out of {np.prod(y_true.shape)}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}")
    
    for i, accuracy in enumerate(accuracy_per_label):
        print(f"Accuracy for label {i}: {accuracy:.2f}")
    
    return y_pred, correctness

def find_optimal_thresholds(y_true, y_scores):
    optimal_thresholds = []
    
    # Assuming y_true and y_scores are 2D arrays with each column representing a label
    for label in range(y_true.shape[1]):  # Loop over columns (labels) not rows
        fpr, tpr, thresholds = roc_curve(y_true[:, label], y_scores[:, label])
        roc_auc = auc(fpr, tpr)
        
        display.append(RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='estimator'))
        # Calculate the distance to the top-left corner for each point on the ROC curve
        
        distances = np.sqrt((1 - tpr) ** 2 + fpr ** 2)
        # Find the smallest distance and its corresponding index
        min_distance_idx = np.argmin(distances)
        optimal_threshold = thresholds[min_distance_idx]
        
        # Handle infinite thresholds, if any
        if np.isinf(optimal_threshold):
            optimal_threshold = np.max(y_scores[:, label])
        
        optimal_thresholds.append(optimal_threshold)
        print(f"Label {label}: Optimal threshold is: {optimal_threshold}")

    return optimal_thresholds

def crop_center(img, cropx, cropy):
    # Assuming img has shape (slices, height, width, channels)
    d, y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    
    # Ensure the cropped image has the same number of channels
    return img[:, starty:starty+cropy, startx:startx+cropx, :]

def load_labels(abnormal_path, acl_path, meniscus_path):
    
  
    labels_abnormal = pd.read_csv(abnormal_path, header=None)
    labels_acl = pd.read_csv(acl_path, header=None)
    labels_meniscus = pd.read_csv(meniscus_path, header=None)

    # Combine labels into a single array
    combined_labels = np.vstack((labels_abnormal.iloc[:,1], labels_acl.iloc[:,1], labels_meniscus.iloc[:,1])).T

    # Debugging: Print label shapes
    print(f"Abnormal Labels Shape: {labels_abnormal.shape}")
    print(f"ACL Labels Shape: {labels_acl.shape}")
    print(f"Meniscus Labels Shape: {labels_meniscus.shape}")
    print(f"Combined Labels Shape: {combined_labels.shape}")

    return combined_labels

def pad_slices(scan, target_slices= 30):
    # Get the current number of slices
    current_slices = scan.shape[0]
    # Check if padding is necessary
    if current_slices < target_slices:
        # Calculate padding amounts
        pad_before = (target_slices - current_slices) // 2
        pad_after = target_slices - current_slices - pad_before
        # Pad the scan with zeros on the slices axis (axis 0)
        padded_scan = np.pad(scan, pad_width=((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
    elif current_slices > target_slices:
        # Calculate the cropping needed
        start = (current_slices - target_slices) // 2
        end = start + target_slices
        # Crop the scan to the target size
        padded_scan = scan[start:end, :, :]
    else:
        # If the number of slices is already equal to the target, no action is needed
        padded_scan = scan
    return padded_scan


def load_data(direc='', target_slices=30):
    # Directories for axial, coronal, and sagittal scans
    directories = [f'{direc}/axial', f'{direc}/coronal', f'{direc}/sagittal']
    data_all_angles = []

    for directory in directories:
        scans = []
        for scan_file in sorted(os.listdir(directory)):
            if scan_file.endswith('.npy'):
                path_to_scan = os.path.join(directory, scan_file)
                scan = np.load(path_to_scan, allow_pickle=True)
                padded_scan = pad_slices(scan, target_slices=target_slices)
                scans.append(padded_scan)

        data_all_angles.append(scans)
        print(f"Loaded {len(scans)} scans from {directory}")

    # Stack the processed data from different angles
    combined_data = [np.stack((axial, coronal, sagittal), axis=-1) 
                     for axial, coronal, sagittal in zip(*data_all_angles)]
    
    return np.array(combined_data)


if __name__ == '__main__':
    main()

