#Knee Injury Detection (KID)
# Deep Knee Injury Detection (DKID) ??

Add a 1-2 line summary of your project here.

## Important Links

| [Timesheet](https://google.com) | [Slack channel](https://google.com) | [Project report](https://google.com) |
|-----------|---------------|-------------------------|


- Timesheet: https://1sfu-my.sharepoint.com/:x:/g/personal/kabhishe_sfu_ca/EZzRYG5ZelJBttJNVGDTbmwBpqNJtml92_OUUOLpUCn4gw?e=o9aVyx
- Slack channel: https://app.slack.com/client/T05JYJAF22G/C05TE8QMLCT/thread/C05TE8QMLCT-1700365574.444619
- Project report: Link your Overleaf project report document.


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.
![alt text](https://github.com/sfu-cmpt340/project_18/blob/main/public/scan.gif?raw=true)

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Demo of 3D CNN prediction (to be updated)....

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```
<a name="demo"></a>
## Demo of 2D image grid processing from .npy 3D MRI volumes



<p align="center">

  <img src="https://github.com/sfu-cmpt340/project_18/blob/main/public/grid_processing.png" width="800" >

</p>
Can be used for any 3D volume data in .npy format in form (slices,height,width) 

- Download 2D CNN pipeline
- Open Image_processing_2D_grids.ipynb or .py

```python
import grid_processing

grid_path_axial = grid_processing.create_grids("MRNet-v1.0_demo/train/axial", 2, 0)
. #repeat for other axes (coronal/sagittal) - or any folders containing .npy volumes
.

#generate triple grids (side by side)
grid_processing.triple_grid(grid_path_axial, grid_path_coronal, grid_path_sagittal)

#generate rgb stacks
grid_processing.rgb_stacks(grid_path_axial, grid_path_coronal, grid_path_sagittal)
    
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone $THISREPO
cd $THISREPO
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
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
- Sign up for [GitHub Education](https://education.github.com/) 
