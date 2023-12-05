import tensorflow as tf
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, GlobalMaxPooling3D
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.layers import add
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, AUC

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.list_physical_devices('GPU')
    data = load_data('../MRNet-v1.0/train')
    labels = load_labels('../MRNet-v1.0/train-abnormal.csv', '../MRNet-v1.0/train-acl.csv', '../MRNet-v1.0/train-meniscus.csv')
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

    # Now 'cropped_data' contains the cropped images
    # You can continue to use 'cropped_data' for further processing such as scaling
    data = cropped_data
    del cropped_data


    # Set your batch size and other parameters
    batch_size = 10
    max_rotations = 3  # This will allow for 0, 90, 180, and 270 degrees of rotation
    flip_prob = 0.5

    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch = data[start:end]

        # Apply augmentation to each slice in each scan
        augmented_batch = np.empty_like(batch)
        for scan_idx, scan in enumerate(batch):
            for slice_idx, img_slice in enumerate(scan):
                # No need to check for channel dimension for MRI data
                rotated = random_rotation(img_slice, max_rotations)
                flipped = horizontal_flip(rotated, flip_prob)
                augmented_batch[scan_idx, slice_idx] = flipped
        
        # Update the original data array with the augmented batch
        data[start:end] = augmented_batch

        print(f"Augmented batch from index {start} to {end}")

    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)
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

    # Check the shape of the datasets
    print('Training set shape:', trainD.shape)
    print('Validation set shape:', valD.shape)
    print('Test set shape:', testD.shape)
    # Assuming 'trainD' and 'trainL' are paths to your data or some lazy loading mechanism
    train_seq = MRISequence(trainD, trainL, batch_size=5)  # Adjust batch_size to a suitable value for your hardware


    # Input layer
    input_shape = (30, 160, 160, 3)
    inputs = Input(shape=input_shape)

    # First Convolutional Block
    x = conv_block(inputs, 32, kernel_size=(7, 7, 7), strides=(2, 2, 2))

    # MaxPooling
    x = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(x)

    # Residual Blocks - adjust the number of blocks and filters based on the successful study
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = Dropout(0.5)(x)

    # Additional Residual Blocks if needed
    # x = residual_block(x, 128)
    # x = Dropout(0.5)(x)
    # ... repeat as necessary ...

    # Transition to the fully connected layer with GlobalMaxPooling
    x = GlobalMaxPooling3D()(x)

    # Fully Connected Layer
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.5)(x)

    # Output Layer
    outputs = Dense(3, activation='sigmoid')(x)  # Assuming 3 classes for the output layer

    # Model definition
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

    # Model Summary
    model.summary()

    # Fit the model with early stopping
    history = model.fit(trainD,
                    trainL,
                    epochs=15, # Adjust the number of epochs as necessary
                    verbose=1, 
                    validation_data= validation,
                    callbacks=[early_stopping])
    
    model.save(os.path.join('models','imageclassifier11.h5'))

    print("Model saved in /models")






def conv_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', batch_norm=True, reg=l2(0.01)):
    conv = Conv3D(filters, kernel_size, padding='same', strides=strides, kernel_regularizer=reg)(x)
    if batch_norm:
        conv = BatchNormalization()(conv)
    if activation:
        conv = Activation(activation)(conv)
    return conv

def residual_block(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu', batch_norm=True, reg=l2(0.01)):
    # Shortcut
    shortcut = x
    if strides != (1, 1, 1) or x.shape[-1] != filters:
        shortcut = Conv3D(filters, (1, 1, 1), padding='same', strides=strides, kernel_regularizer=reg)(x)
        if batch_norm:
            shortcut = BatchNormalization()(shortcut)

    # Residual path
    x = conv_block(x, filters, kernel_size, strides, activation, batch_norm, reg)
    x = conv_block(x, filters, kernel_size, activation=None, batch_norm=batch_norm, reg=reg)

    # Add shortcut value to main path
    x = add([shortcut, x])
    if activation:
        x = Activation(activation)(x)
    return x

class MRISequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array(batch_x), np.array(batch_y)

def random_rotation(img, max_rotations):
    # Rotate image by a random number of 90-degree steps
    k = random.randint(0, max_rotations)  # Choose a random rotation
    return np.rot90(img, k=k, axes=(0, 1))  # Rotate on the (height, width) plane

def horizontal_flip(img, flip_prob):
    # Flip image horizontally with a given probability
    if random.random() < flip_prob:
        return np.flip(img, axis=1)  # Flip on the width axis
    return img

def crop_center(img, cropx, cropy):
    # Assuming img has shape (slices, height, width, channels)
    d, y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    
    # Ensure the cropped image has the same number of channels
    return img[:, starty:starty+cropy, startx:startx+cropx, :]

def display_data(data, labels):
    # Display the first image of each angle from the combined data along with the labels
    for i in range(len(data)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # The data shape is expected to be (slices, 255, 255, angles)
        # We will display the middle slice for each angle.
        middle_slice_index = data[i].shape[0] // 2

        for j, title in enumerate(['Axial', 'Coronal', 'Sagittal']):
            # Extract the middle slice for the current angle
            image = data[i][middle_slice_index, :, :, j]
            
            axs[j].imshow(image, cmap='gray')
            axs[j].axis('off')  # Turn off axis
            axs[j].title.set_text(title)

        label_names = ['Abnormal', 'ACL', 'Meniscus']
        label_str = ', '.join(f'{name}: {value}' for name, value in zip(label_names, labels[i]))
        plt.suptitle(f'Labels: {label_str}')

        # Show the plot
        plt.show()

        # Break after the first set of images for brevity
        if i == 1:
            break



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
