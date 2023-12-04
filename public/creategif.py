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
 
def main():
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


    batch_min = all.min()
    batch_max = all.max()
    all = (all.astype(np.float32) - batch_min) / (batch_max - batch_min)

    
    model_path = os.path.join('models', 'imageclassifier5.h5')
    model = load_model(model_path)
    prediction = model.predict(all, verbose=0)
    prediction = prediction[0]
    abnormal_pred = prediction[0]
    acl_pred = prediction[1]
    meniscus_pred = prediction[2]


    if acl_pred > 0.55:
        print("ACL injury")
        print("type ", type(acl_pred))
    elif abnormal_pred > 0.55:
        print("No ACL injury, but has abnormalities")
        
    else:
        print("No ACL injury")
    
    print(prediction)


    # duration is the number of milliseconds between frames; this is 40 frames per second
    im[0].save("public/scan.gif", save_all=True, append_images=im[1:], duration=50, loop=0)


def crop_center(img, cropx, cropy):
    # Assuming img has shape (slices, height, width, channels)
    d, y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    
    # Ensure the cropped image has the same number of channels
    return img[:, starty:starty+cropy, startx:startx+cropx, :]

def pad_slices(scan, target_slices= 50):
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
