import re
import cv2 as cv
import torch
import torchvision
import os
from PIL import Image
import numpy as np
import pandas as pd
import math
labels  = ['A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Others', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y','DD']
from PIL import Image
class SampleDataset(torch.utils.data.Dataset):
    def __init__(self,root, type,transforms):
        self.trans = transforms
        self.root = root
        self.type = type
        self.list_imgs = []
        self.list_labels = []
        self.classes = open('labels.txt', 'r', encoding = 'utf-8').read().splitlines()
        import glob

        self.root = "/data/disk1/congvu/Face_Attr/Age_Prediction/Classification-Models/OCR_dataset"
        if type == 'train':
                file = open('val.txt','r').read().splitlines()
                for line in file:
                            img, label = line.split("| ")
                            self.list_imgs.append(img) 
                            self.list_labels.append(int(label))
        else:
            file = open('val.txt','r').read().splitlines()
            for line in file:
                            img, label = line.split("| ")
                            self.list_imgs.append(img) 
                            self.list_labels.append(int(label))

    

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, idx):
        if self.type == 'train':
            img=Image.open(self.list_imgs[idx])
        else:
            img=Image.open(self.list_imgs[idx]).convert("RGB")
        if self.trans:
          img = self.trans(img)
        label=self.list_labels[idx]
        return img,label
    
def get_dataloader_train( root,type,trans,batch_size, shuffle=False):
    ds_avatarsearch = SampleDataset(root, type,trans)

    # Use dataloader with num_workers is 0 (not use num_workers)
    dataloader = torch.utils.data.DataLoader(ds_avatarsearch, batch_size=batch_size, 
                                             shuffle=True)
    
    return dataloader




