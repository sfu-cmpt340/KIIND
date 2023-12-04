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
 
def main():
    threshold = [0.6039676070213318, 0.43964335322380066, 0.46997469663619995]
    injuryType = sys.argv[1]
    im = []
    image_array = np.load('uploads/data_a.npy')
    image_array_s = np.load('uploads/data_s.npy')
    image_array_c = np.load('uploads/data_c.npy')

    for i in range(len(image_array)):
        im.append(Image.fromarray(image_array[i].astype('uint8')))
    for i in range(len(image_array_s)):
        im.append(Image.fromarray(image_array_s[i].astype('uint8')))
    for i in range(len(image_array_c)):
        im.append(Image.fromarray(image_array_c[i].astype('uint8')))
        

    array_a = pad_slices(image_array)
    array_s = pad_slices(image_array_s)
    array_c = pad_slices(image_array_c)


    all = np.stack((array_a, array_s, array_c), axis=-1)
    all = np.array([crop_center(all, 160, 160) ])

    max_rotations = 3  # This will allow for 0, 90, 180, and 270 degrees of rotation
    flip_prob = 0.5

    max_rotations = 3  # This will allow for 0, 90, 180, and 270 degrees of rotation
    flip_prob = 0.5
    batch = all

    augmented_batch = np.empty_like(batch)
    for scan_idx, scan in enumerate(batch):
        for slice_idx, img_slice in enumerate(scan):
            # No need to check for channel dimension for MRI data
            rotated = random_rotation(img_slice, max_rotations)
            flipped = horizontal_flip(rotated, flip_prob)
            augmented_batch[scan_idx, slice_idx] = flipped

    # Update the original data array with the augmented batch
    all = augmented_batch
    
    model_path = os.path.join('models', 'imageclassifier11.h5')
    model = load_model(model_path)
    prediction = model.predict(all, verbose=0)
    prediction = prediction[0]
    abnormal_pred = prediction[0]
    acl_pred = prediction[1]
    meniscus_pred = prediction[2]


    if injuryType == 'acl':
        if acl_pred >= threshold[1]:
            print("ACL injury")
            # print("type ", type(acl_pred))
        elif abnormal_pred >= threshold[0]:
            print("No ACL injury, but has abnormalities")
            
        else:
            print("No ACL injury")
    elif injuryType == 'meniscus':
        if meniscus_pred >= threshold[2]:
            print("Meniscus injury")
            # print("type ", type(acl_pred))
        elif abnormal_pred >= threshold[0]:
            print("No meniscus injury, but has abnormalities")
            
        else:
            print("No meniscus injury")
    
    print(prediction)


    # duration is the number of milliseconds between frames; this is 40 frames per second
    im[0].save("public/scan.gif", save_all=True, append_images=im[1:], duration=50, loop=0)


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


def main2():

    print(type(sys.argv[1]))
    print(type('.argv[1]'))


    
    # print(gifImage)
    # obj = json.loads(sys.argv[1])
    datapath = str(os.path.dirname(os.path.realpath(__file__)))+'/file.txt'
    f = open(datapath,'r')

    base64data = f.read()
    # print(base64data)

    npArry = base64.b64decode(base64data)

    print(len(npArry))
    # image = Image.open(io.BytesIO(npArry[0]))
    # image_np = np.array(image)
    # print(image_np)


    


if __name__ == '__main__':
    main()
