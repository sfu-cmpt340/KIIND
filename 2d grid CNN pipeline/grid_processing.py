

import os
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

#this function comes from https://stackoverflow.com/a/46877433
def pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid 

#these functions were created by myself with the help of ChatGPT
def create_grids(folder_path_images, crop_factor, max_grids):
    ''' 
    @folder_path_images: (str) path to images (.npy)
    @crop_factor: (int) 1/crop_factor to be removed from the height and width
    @max_grids: (0 or 1) (0 for 4x4 grids, 1 for max size grids)

    returns path to folder with grid images, will be named with whatever comes before the .npy
    '''
    # folder_path_images = 'MRNet-v1.0/valid/axial'  # Update this with the path to your folder containing .npy files
    file_names = sorted(os.listdir(folder_path_images))
    file_paths = [os.path.join(folder_path_images, file_name) for file_name in file_names if file_name.endswith('.npy')]


    #Step 2 get current path and create new folder to save data in current path
    path = os.path.join(os.getcwd(), folder_path_images + '_grid_imagesTESTING')

    if os.path.isdir(path):
        print("Folder: {", os.path.basename(path), "} already exists! Create a new folder name or images will be put into pre-existing folder!")
    else:
        os.makedirs(path)
        print("grids data will be saved in:")
        print(path)


    pbar = tqdm(total=len(file_paths), desc = "Creating Grid Images", colour='#669bbc')
    count = 0

    for file_path in file_paths:
        
        #load volumes
        vol = np.load(file_path)
        
        nSlices = np.size(vol,0)#num of slices in image
        volName = os.path.basename(file_path) #name will be same for all axes (just finding the label 0000, 0001 etc.)
        end = volName.find('.')
        volName = volName[:end]  

        # print(f"File Name: {os.path.basename(file_path)}")
            
        #add all slices into a single array "images", a PIL, Image objecs
        slices= []
        for i in range(nSlices):
            image = Image.fromarray(vol[i,:,:])
            # image = image.resize((128,128), 1)# can resize images to make smaller and reduce compute
            w, h = image.size
            image = image.crop((w//(crop_factor*2),h//(crop_factor*2) , w-w//(crop_factor*2), h-h//(crop_factor*2)))
            slices.append(image)
            
            
        if max_grids == 0: #4x4 grids will be created
            n=4
            nSlices_Excluded = nSlices - n**2
        else: 
            #if want to create largest possible grids  
            gridEdgeSize = int(np.sqrt(nSlices)) #will round down to smallest smaller edge size for a squrae grid
            nSlicesInGrid = gridEdgeSize**2 #number of slices to include in grid to ensure square
            nSlices_Excluded = nSlices-nSlicesInGrid #number of slices excluded to ensure perfect square (rounding to nearest square)
            n = gridEdgeSize
        
        #divide the number of exluded slices into 2 to remove from beginning and end, 
        #to hopefully capture more information in middle slices
        nSlices_Excluded_first_half = nSlices_Excluded//2
        nSlices_Excluded_second_half = nSlices_Excluded-nSlices_Excluded_first_half
        start = nSlices_Excluded_first_half
        end = nSlices-nSlices_Excluded_second_half


        if max_grids ==1:
            grid = pil_grid(slices[start:end],n)
            grid = grid.resize((512,512), 1)

        else:
            grid = pil_grid(slices[start:end],n)

        # print("File Name: {} | total slices: {}, slices included: {}".format(os.path.basename(file_path),nSlices,len(slices[start:end]) ))
        
        #depending on the values at position intVolName on the csvs (info stored in dataArray) place image in correct folder
        grid.save(os.path.join(path, 'grid_{}.jpeg'.format(volName)))
        count = count+1
        pbar.update(1)
        
    pbar.close()
    print("\n===================DONE=======================")
    print("Num grid Images created:",count)


    return path

def triple_grid(data_dir1, data_dir2, data_dir3):

    #get folder containing grids to save triple grids in 
    path = os.path.join(os.path.dirname(data_dir1), 'triple_grids')

    #create folder to save triple grids (if already exists will save it there)
    if os.path.isdir(path):
        print("Folder: {", os.path.basename(path), "} already exists! Create a new folder name or images will be put into pre-existing folder!")
    else:
        os.makedirs(path)
        print("Data will be saved in:")
        print(path)
    
    #ensure only .jpegs are being processed
    file_names1 = sorted(os.listdir(data_dir1))
    file_paths_axial = [os.path.join(data_dir1, file_name) for file_name in file_names1 if file_name.endswith('.jpeg')]

    file_names2 = sorted(os.listdir(data_dir2))
    file_paths_coronal = [os.path.join(data_dir2, file_name) for file_name in file_names2 if file_name.endswith('.jpeg')]

    file_names3 = sorted(os.listdir(data_dir3))
    file_paths_sagittal = [os.path.join(data_dir3, file_name) for file_name in file_names3 if file_name.endswith('.jpeg')]

    count = 0
    pbar = tqdm(total=len(file_paths_axial), desc = "Combining Grid Images", colour='#669bbc')
    for file_path_axial, file_path_coronal, file_path_sagittal in zip(file_paths_axial, file_paths_coronal, file_paths_sagittal):

        axial_img = Image.open(file_path_axial)
        coronal_img = Image.open(file_path_coronal)
        sagittal_img = Image.open(file_path_sagittal)
        
        imName = os.path.basename(file_path_axial) #can be any, just want the first 4 digits
        end = imName.find('.')
        # start = len(volName[:end])-4
        imName = imName[len(imName[:end])-4:end]
        # print("volNAME",imName)
        
        grid = pil_grid([axial_img,coronal_img,sagittal_img])
        
        grid.save(os.path.join(path, 'tripleGrid_{}.jpeg'.format(imName)))
        pbar.update(1)
        count = count+1
    

    pbar.close()
    print("\n===================DONE=======================")
    print("Num triple grid Images created:",count)

def rgb_stacks(data_dir1, data_dir2, data_dir3):

    #get folder containing grids to save triple grids in 
    path = os.path.join(os.path.dirname(data_dir1), 'rgb_stacks')

    #create folder to save triple grids (if already exists will save it there)
    if os.path.isdir(path):
        print("Folder: {", os.path.basename(path), "} already exists! Create a new folder name or images will be put into pre-existing folder!")
    else:
        os.makedirs(path)
        print("Data will be saved in:")
        print(path)
        
    #ensure only .jpegs are being processed
    file_names1 = sorted(os.listdir(data_dir1))
    file_paths_axial = [os.path.join(data_dir1, file_name) for file_name in file_names1 if file_name.endswith('.jpeg')]

    file_names2 = sorted(os.listdir(data_dir2))
    file_paths_coronal = [os.path.join(data_dir2, file_name) for file_name in file_names2 if file_name.endswith('.jpeg')]

    file_names3 = sorted(os.listdir(data_dir3))
    file_paths_sagittal = [os.path.join(data_dir3, file_name) for file_name in file_names3 if file_name.endswith('.jpeg')]


    count = 0
    pbar = tqdm(total=len(data_dir1), desc = "Combining Grid Images", colour='#669bbc')
    for file_path_axial, file_path_coronal, file_path_sagittal in zip(file_paths_axial, file_paths_coronal, file_paths_sagittal):

        axial_img = Image.open(file_path_axial)
        coronal_img = Image.open(file_path_coronal)
        sagittal_img = Image.open(file_path_sagittal)
        
        #convert images to single channel to then combine into rgb (uint8)
        im1_s = axial_img.convert('L') #s for single channel 
        im2_s = coronal_img.convert('L')
        im3_s = sagittal_img.convert('L')
        rgb = np.dstack((np.array(im1_s),np.array(im2_s),np.array(im3_s)))
        rgb_im = Image.fromarray(rgb)
        
        imName = os.path.basename(file_path_axial) #can be any, just want the first 4 digits
        end = imName.find('.')
        imName = imName[len(imName[:end])-4:end]
        # print("volNAME",imName)
        
        
        rgb_im.save(os.path.join(path, 'rgbStack_{}.jpeg'.format(imName)))
        pbar.update(1)
        count = count+1
    

    pbar.close()
    print("\n===================DONE=======================")
    print("Num rgb triple channel Images created:",count)