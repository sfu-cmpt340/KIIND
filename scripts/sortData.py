import csv
import os
import shutil

# there will be overlap as data can be all of acl/meniscus/abnormal

dataArray = [[], #number
            [], #acl
            [], #meniscus
            []] #abnormal
count = 0

# acl
with open('data/MRNet-v1.0/train-acl.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        if len(row) != 0:
            dataArray[0].append(row[0]) 
            dataArray[1].append(int(row[1])) 
            count += 1

# meniscus
count = 0
with open('data/MRNet-v1.0/train-meniscus.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        if len(row) != 0:
            dataArray[2].append(int(row[1])) 
            count += 1

# abnormal
count = 0
with open('data/MRNet-v1.0/train-meniscus.csv', newline='') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in csv_reader:
        if len(row) != 0:
            dataArray[3].append(int(row[1]))
            count += 1


print(dataArray[1][1])

print(str(dataArray[1][1]).zfill(4))

for i in range(1130):
    src_path_a = os.path.join('data/MRNet-v1.0/train/axial', str(i).zfill(4)+'.npy')
    src_path_c = os.path.join('data/MRNet-v1.0/train/coronal', str(i).zfill(4)+'.npy')
    src_path_s = os.path.join('data/MRNet-v1.0/train/sagittal', str(i).zfill(4)+'.npy')
    
    # if (dataArray[1][i] == 1):
    #     # print(dataArray[1][i])
    #     dst_path_a = os.path.join('data/MRNet-v1.0/train2/axial/healthy', i+'.npy')
    #     shutil.copy(src_path_a, dst_path_a)
    #acl
    if (dataArray[1][i] == 1):
        dst_path_a = os.path.join('data/MRNet-v1.0/train2/axial/acl', str(i).zfill(4)+'.npy')
        dst_path_c = os.path.join('data/MRNet-v1.0/train2/coronal/acl', str(i).zfill(4)+'.npy')
        dst_path_s = os.path.join('data/MRNet-v1.0/train2/sagittal/acl', str(i).zfill(4)+'.npy')

        shutil.copy(src_path_a, dst_path_a)
        shutil.copy(src_path_c, dst_path_c)
        shutil.copy(src_path_s, dst_path_s)
    #meniscus
    if (dataArray[2][i] == 1):
        dst_path_a = os.path.join('data/MRNet-v1.0/train2/axial/meniscus', str(i).zfill(4)+'.npy')
        dst_path_c = os.path.join('data/MRNet-v1.0/train2/coronal/meniscus', str(i).zfill(4)+'.npy')
        dst_path_s = os.path.join('data/MRNet-v1.0/train2/sagittal/meniscus', str(i).zfill(4)+'.npy')

        shutil.copy(src_path_a, dst_path_a)
        shutil.copy(src_path_c, dst_path_c)
        shutil.copy(src_path_s, dst_path_s)
        
    #abnormal
    if (dataArray[3][i] == 1):
        dst_path_a = os.path.join('data/MRNet-v1.0/train2/axial/abnormal', str(i).zfill(4)+'.npy')
        dst_path_c = os.path.join('data/MRNet-v1.0/train2/coronal/abnormal', str(i).zfill(4)+'.npy')
        dst_path_s = os.path.join('data/MRNet-v1.0/train2/sagittal/abnormal', str(i).zfill(4)+'.npy')

        shutil.copy(src_path_a, dst_path_a)
        shutil.copy(src_path_c, dst_path_c)
        shutil.copy(src_path_s, dst_path_s)

    if (dataArray[1][i] == 0 and dataArray[2][i] == 0 and dataArray[3][i] == 0):
        dst_path_a = os.path.join('data/MRNet-v1.0/train2/axial/healthy', str(i).zfill(4)+'.npy')
        dst_path_c = os.path.join('data/MRNet-v1.0/train2/coronal/healthy', str(i).zfill(4)+'.npy')
        dst_path_s = os.path.join('data/MRNet-v1.0/train2/sagittal/healthy', str(i).zfill(4)+'.npy')

        shutil.copy(src_path_a, dst_path_a)
        shutil.copy(src_path_c, dst_path_c)
        shutil.copy(src_path_s, dst_path_s)