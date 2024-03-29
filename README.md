<!-- #Knee Injury Detection (KID) -->
# Knee Injury Neural Network Detector (KINND)
### CMPT 340 - Biomedical Computing

We have developed a Convolutional Neural Network (CNN) model designed to analyze knee Magnetic Resonance Imaging (MRI) scans, earning us an A+ for our term research project. The model provides outputs indicating the presence of an Anterior Cruciate Ligament (ACL) injury, a meniscus injury, an abnormality, or the absence of any injuries/abnormalities.

## Important Links
<!-- 
| [Timesheet](https://google.com) | [Slack channel](https://google.com) | [Project report](https://google.com) |
|-----------|---------------|-------------------------| -->


- Timesheet: https://1sfu-my.sharepoint.com/:x:/g/personal/kabhishe_sfu_ca/EZzRYG5ZelJBttJNVGDTbmwBpqNJtml92_OUUOLpUCn4gw?e=o9aVyx
- Slack channel: https://app.slack.com/client/T05JYJAF22G/C05TE8QMLCT/thread/C05TE8QMLCT-1700365574.444619
- Project report: https://www.overleaf.com/project/650ca31f1a3e9fd765a19394
- Website: https://cmpt340-project-758b976dd842.herokuapp.com/
- Demo video: https://www.youtube.com/watch?v=40N3V82554M


## Video/demo/GIF

<img style="align:center" src="https://github.com/sfu-cmpt340/project_18/blob/main/public/scan.gif" width="400" >

[Watch the demo](https://www.youtube.com/watch?v=40N3V82554M)



## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

<!-- 4. [Guidance](#guide) -->


<a name="demo"></a>
## 1. Demo of 3D CNN prediction

```python
import tensorflow as tf
from tensorflow.keras.models import load_model

model_path = os.path.join('models', 'imageclassifier11.h5')
model = load_model(model_path)
prediction = model.predict(data)
```
<a name="demo"></a>
## Demo of 2D image grid processing from .npy 3D MRI volumes



<p align="center">

  <img src="https://github.com/sfu-cmpt340/project_18/blob/main/public/grid_processing.png" width="800" >

</p>
Can be used for any 3D volume data in .npy format in form (slices,height,width)
Required packages: numpy, PIL, shutil, tqdm, os, collections

- Download 2D CNN pipeline
- Open Image_processing_2D_grids.ipynb or .py

```python
import grid_processing

grid_path_axial = grid_processing.create_grids("MRNet-v1.0_demo/train/axial", 2, 0)
. #repeat for other axes (coronal/sagittal) - or any folders containing .npy volumes
.

#generate triple grids (side by side)
triple_grids_path = grid_processing.triple_grid(grid_path_axial, grid_path_coronal, grid_path_sagittal)

#generate rgb stacks
rgb_stacks_path = grid_processing.rgb_stacks(grid_path_axial, grid_path_coronal, grid_path_sagittal)

#create balanced dataset
temp,valid_labels = np.loadtxt("MRNet-v1.0_demo/valid-acl.csv", #can add or remove more temps if there are more columns
                 delimiter=",", dtype=int, unpack=True)
bal_imgs_path_train, bal_labels_train = grid_processing.balance_dataset(rgb_stacks_path,train_labels,'balanced_acl_rgb_stacks')

    
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── 2d grid CNN pipeline         ## 2D collage/grid CNN and related functions
├── models                       ## the model hooked up to the website
├── models2                      ## other models
├── MRI Model Builder            ## model building scripts/notebooks
├── MRNet-v1.0                   ## some sample data to test the website with
├── public                       ## stylesheets, images
├── scripts                      ## additional scripts
├── uploads                      ## stores the uploaded data from the website
├── views                        ## html pages
├── README.md                    ## You are here
├── requirements.txt             ## python libraries to run the script on the website
├── requirements2.txt            ## python libraries to run the model training/evaluating scripts


```

<a name="installation"></a>

## 2. Installation

No need to install anything, simply go to https://cmpt340-project-758b976dd842.herokuapp.com/ 

There are some sample test data in /MRNet-v1.0 that you can upload onto the website

However, if you would like to run the app locally:
First make sure you have Node.js installed (https://nodejs.org/en/download)
```bash
git clone git@github.com:sfu-cmpt340/project_18.git
cd project_18
pip install -r requirements.txt
npm install
node index.js
```
Then go to http://localhost:3000/

<a name="repro"></a>
## 3. Reproduction
First request the dataset from https://stanfordmlgroup.github.io/competitions/mrnet/ and store it (folder will be named 'MRNet-v1.0', so replace the current folder named that with this one) inside the project folder (we aren't able to include too many datasets due to the Stanford University School of Medicine MRNet Dataset Research Use Agreement)

Install requirements:
```bash
pip install -r requirements2.txt
```

Make the model:
```bash
$ python scripts/makeModel.py
...
Model saved in /models
```

Evaluate the model:
```bash
$ python scripts/evaluateModel.py
...
Precision: 0.6810, Recall: 0.7603, Accuracy: 0.7434, AUC: 0.8147
```

Using the model to predict:
```bash
$ python scripts/runModel.py
...
Predictions saved to predictions.csv
```


<!-- <a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)  -->
