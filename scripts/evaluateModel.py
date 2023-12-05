from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, AUC
import os
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np


def main():
    data = load_data('../MRNet-v1.0/train')
    labels = load_labels('../MRNet-v1.0/train-abnormal.csv', '../MRNet-v1.0/train-acl.csv', '../MRNet-v1.0/train-meniscus.csv')


    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
    total_size = data.shape[0]

    # Assuming data and labels are numpy arrays
    total_size = data.shape[0]

    # Calculate the indices for splitting
    train_end = int(train_size)
    val_end = train_end + int(val_size)

    # Split the data
    trainD = data[:train_end]
    valD = data[train_end:val_end]
    testD = data[val_end:]

    # Split the labels
    trainL = labels[:train_end]
    valL = labels[train_end:val_end]
    testL = labels[val_end:]

    # Create tuples for easy handling
    train = (trainD, trainL)
    validation = (valD, valL)
    test = (testD, testL)


    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    auc = AUC()

    batch_size = 5

    model_path = os.path.join('../models/imageclassifier11.h5')
    model = load_model(model_path)

    for i in range(0, len(test[0]), batch_size):  # test[0] is the data component of the test set
        # Create a batch of data and labels
        X, y = test[0][i:i+batch_size], test[1][i:i+batch_size]
        
        # Make predictions
        yhat = model.predict(X)
        
        # Update metrics
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
        auc.update_state(y, yhat)


    # Calculate final results
    precision_result = pre.result().numpy()
    recall_result = re.result().numpy()
    accuracy_result = acc.result().numpy()
    auc_result = auc.result().numpy()

    # Print results
    print(f'Precision: {precision_result:.4f}, Recall: {recall_result:.4f}, Accuracy: {accuracy_result:.4f}, AUC: {auc_result:.4f}')



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
