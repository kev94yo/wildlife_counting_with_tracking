import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

import sys
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

import collections
import gc
import json
import os
from os import listdir
from os.path import isfile, join
import random
import time
import warnings
warnings.simplefilter("ignore")

from albumentations import *
from albumentations.pytorch import ToTensor
import cv2
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import tifffile as tiff
import timm
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, sampler
from tqdm import tqdm_notebook as tqdm
from centroid_tracker import CentroidTracker
# test_tracker = CentroidTracker()
# print(test_tracker.maxDisappeared)

DATASET = "../data/"
CROPPED_DATA = "../data/"

TRAIN_CROPPED_DATA = "../data/crop_train/"
TEST_CROPPED_DATA = "../data/crop_test/"

BATCH_SIZE = 32
EPOCHS = 300
NUM_WORKERS = 4
SEED = 2021
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_DATA_PATH = "../data/train/"
TEST_DATA_PATH = "../data/test/"
with open('../data/metadata/iwildcam2021_megadetector_results.json', encoding='utf-8') as json_file:
    megadetector_results = json.load(json_file)

megadetector_results_df = pd.DataFrame(megadetector_results["images"])

def Convert(lst):
    res_dct = {lst[i]: True for i in range(0, len(lst))}
    return res_dct

photo_id_dict = Convert(megadetector_results_df['id'].to_list())
print('8d705d8a-21bc-11ea-a13a-137349068a90' in photo_id_dict)

with open('../data/metadata/iwildcam2021_test_information.json', encoding='utf-8') as json_file:
    test_information =json.load(json_file)
    
df_test_info = pd.DataFrame(test_information["images"])[["id", "seq_id"]]

mypath = '../data/test/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
seq_id_dict = {}
for id in onlyfiles:
    seq_id = df_test_info[df_test_info["id"] == id[:-4]]['seq_id'].item()
    
    if seq_id in seq_id_dict:
        seq_id_dict[seq_id].append(id)
    else:
        seq_id_dict[seq_id] = [id]

def set_seed(seed=2**3):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
set_seed(SEED)

# ====================================================
# EfficientNet Model
# ====================================================

class enet_v2(nn.Module):

    def __init__(self, backbone, out_dim, pretrained=False):
        super(enet_v2, self).__init__()
        self.enet = timm.create_model(backbone, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def forward(self, x):
        x = self.enet(x)
        x = self.myfc(x)
        return x
    
model = enet_v2(backbone="tf_efficientnet_b0", out_dim=205)
model.to(DEVICE)
model.load_state_dict(torch.load(f"../{EPOCHS}_.pth"))
model.eval()

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

mean = np.array([0.37087523, 0.370876, 0.3708759] )
std = np.array([0.21022698, 0.21022713, 0.21022706])

class IWildcamTestDataset(Dataset):
    def __init__(self, image, tfms=None):
        # self.ids = df["id"]
        # self.idx = df["idx"]
        self.tfms = tfms
        self.image = image
        
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        # size = (256, 256)
        # image_id = self.ids[idx]
        # image_idx = self.idx[idx]
        
        cropped_image = np.array(self.image)
        
        cropped_image = cv2.resize(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB), (256, 256))
        

        if self.tfms is not None:
            augmented = self.tfms(image=cropped_image)
            cropped_image = augmented['image']
            
        # we should normalize here
        return img2tensor((cropped_image/255.0 - mean)/std)



count_results = []
seq_number = 1
print("start")
for seq_id in seq_id_dict:
    print(seq_number, seq_id)
    seq_number += 1
    ct = CentroidTracker()
    size = (480,270)
    # fig = plt.figure(figsize = (25,16))
    for i, item in enumerate(seq_id_dict[seq_id]):
        print("    " + item)
        if item[:-4] not in photo_id_dict:
            continue
        detections = megadetector_results_df[megadetector_results_df['id'] == item[:-4]]['detections'].item()
        print("    ", detections)
        rects = []
        class_results = []

        # first, draw bbox with cv
        im_path = TEST_DATA_PATH + item
        # ax = fig.add_subplot(4, 3, i+1, xticks=[], yticks=[])
        im = Image.open(im_path)
        im = im.resize(size)
        # draw = ImageDraw.Draw(im)

        for detection in detections:
            if detection['conf'] <= 0.8:
                # print("Detection is shit, so skip")
                continue
            
            xmin, ymin, width, height = detection['bbox']
            xmax = xmin + width
            ymax = ymin + height
            
            imageWidth= im.size[0]
            imageHeight= im.size[1]

            x1, x2, y1, y2 = xmin * imageWidth, xmax * imageWidth, ymin * imageHeight, ymax * imageHeight
            
            rects.append((x1, y1, x2, y2))
            
            # draw.line([(x1, y1), (x1, y2), (x2, y2),
            #        (x2, y1), (x1, y1)], width=4, fill='Red')
            
            cropped_image = im.crop((x1, y1, x2, y2)).resize((256, 256))
            ds_test = IWildcamTestDataset(cropped_image)
            dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
           
            

            with torch.no_grad():
                for img in tqdm(dl_test):
                    img = img.to(DEVICE)
                    outputs = model(img)
                    output_labels = torch.argmax(outputs, dim=1).tolist()
                    # print(output_labels)
                    class_results.append(output_labels[0])

        objects = ct.update(rects, class_results)
        # loop over the tracked objects
        # for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            
            # text = "ID {}".format(objectID)
            # draw.text((centroid[0] - 10, centroid[1] - 10), text)


        # plt.imshow(im)
    count_results.append([seq_id] + ct.class_count[1:])

sub = pd.read_csv("../data/sample_submission.csv")
col_Predicted = [col for col in sub.columns if "Predicted" in col]
col_Predicted.insert(0, "Id")

df = pd.DataFrame(count_results, columns=col_Predicted)
df.to_csv("submission10.csv", index=False)