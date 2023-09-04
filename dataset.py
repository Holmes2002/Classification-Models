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
        self.classes = ["Male","Female"]
        for fol in os.listdir(f'{self.root}/{type}'):
            for file in os.listdir(f'{self.root}/{type}/{fol}'):
                if file in ['toc-dai-cho-nam-dap-xu.png', 'kieu-toc-nam-dai-18.jpeg', 'csfdfsd.png', 'toc-bui-nam-15.png']:
                    continue
                if file in ['fgffgdgf (3).png', 'fgffgdgf (2).png', 'fgffgdgf (1).png', 'dffsdsfd (1).jpg', 'fgffgdgf (7).png', 'fgffgdgf (34).jpg', 'csfdfsd.png', 'vvcxcxvvcx (15).jpg', 'sfddfsfsd (4).png', 'cxvvcxvcxxcv (3).png', 'cxvvcxvcxxcv (5).png', 'cxvvcxvcxxcv (6).png', 'cvcvc (4).png', '000219.jpg', 'cxvvcxvcxxcv (7).png', 'fgffgdgf (14).jpg', 'cxvvcxvcxxcv (65).jpg', 'cxvvcxvcxxcv (1).png', 'sfddfsfsd (5).png', 'cxvvcxvcxxcv (2).png', 'sfdfdssfdsfd (3).png', 'vvcxcxvvcx (3).png', 'fdsfdsdsfsfd (74).jpg', '000039.jpg', 'xvcvcxcvx (59).jpg', 'cvcvc (3).png', 'cxvvcxvcxxcv (52).jpg', 'cvcvc (2).png', 'vvcxcxvvcx (1).png', 'fdfsdvccvc (141).jpg', 'cxvvcxvcxxcv (4).png']:
                    continue
                self.list_imgs.append(f'{self.root}/{type}/{fol}/{file}')
                self.list_labels.append(self.classes.index(fol))


      
    

    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, idx):
        img=Image.open(self.list_imgs[idx])
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




