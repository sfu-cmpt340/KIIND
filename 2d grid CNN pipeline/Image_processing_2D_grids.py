# %% [markdown]
# <font size=5> Install dependencies (if not already present)

# %%
#install dependencies if don't already have
!pip install tqdm -U
!pip install PIL -U
!pip install numpy -U
!pip

# %%
import grid_processing
import os

# %% [markdown]
# <font size=5> Create Grids (use MRNet-v1.0_demo for an example with few images, instead of entire dataset)

# %%
#will be saved in folder containing images used to create grids (.npy files) 
#name will be grid_nameOfVolume.jpeg

#parameters of create_grids:
#file path path to file
#crop factor - 2 = slices all 1/2 smaller, 3 = slices all 1/3 smaller etc.
#max_grids = 0 grids will be made of 4x4 slices 
#max_grids = 1 grids will be made of largest possible number of slices (perfect square) and resized to 512x512

#returns path to saved grid imags


#axial grid images
grid_path_axial = grid_processing.create_grids("MRNet-v1.0_demo/valid/axial", 2, 0)

# %%
#coronal grid images 
grid_path_coronal = grid_processing.create_grids("MRNet-v1.0_demo/valid/coronal", 2, 0)

# %%
#sagittal grid images 
grid_path_sagittal = grid_processing.create_grids("MRNet-v1.0_demo/valid/sagittal", 2,0)

# %% [markdown]
# <font size=5> Create triple grids

# %%
#will save triple grids in folder containing the 3 grid folders created above
#parameters: filepath1, filepath2, filepath3 containing jpegs created above - will combine them into triple grids
grid_processing.triple_grid(grid_path_axial, grid_path_coronal, grid_path_sagittal)

# %% [markdown]
# <font size=5> Create rgb stacks

# %%
#will save rgb stacks of axial, coronal, sagittal images (putting each into a channel (R G B))
#will save triple grids in folder containing the 3 grid folders created above
# parameters: filepath1, filepath2, filepath3 containing jpegs created above - will combine them into rgb stacks
grid_processing.rgb_stacks(grid_path_axial, grid_path_coronal, grid_path_sagittal)

# %%



