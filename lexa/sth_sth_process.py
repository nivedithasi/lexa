import av
import torch
import numpy as np
import os
import h5py
import cv2
import glob

import os
import torchvision
from transforms_video import ComposeMix, RandomCropVideo, RandomRotationVideo, Scale
import json
import numpy as np
import torchvision
from transforms_video import *
from tensorflow import *
import torchvision
from transforms_video import *
import tensorflow as tf
import pdb
import time

from tensorflow.keras.layers import Permute 
from collections import namedtuple, defaultdict, Counter
import json

ListData = namedtuple('ListData', ['id', 'label', 'path'])
FRAMERATE = 12  # default value
        
class VideoFolder():

    def __init__(self, root):
        self.root = root
            
    def crop(self, img, crop_wd, crop_ht):
        ht, wd, _ = img.shape
        if crop_wd > wd or crop_ht > ht:
            padding = [
                    (crop_wd - wd) // 2 if crop_wd > wd else 0,
                    (crop_ht - ht) // 2 if crop_ht > ht else 0,
                    (crop_wd - wd + 1) // 2 if crop_wd > wd else 0,
                    (crop_ht - ht + 1) // 2 if crop_ht > ht else 0,
                    ]
            img = pad(img, padding, fill=0)
            ht, wd, _ = img.shape
            if crop_wd == wd and crop_ht == ht:
                return img
        crop_top = int(round((ht - crop_ht) / 2.0))
        crop_left = int(round((wd - crop_wd) / 2.0))
        return img[crop_top:crop_top+crop_wd, crop_left:crop_left+crop_ht]


    def resize(self, img, size):
        h, w, _= img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, (ow, oh), interpolation=cv2.INTER_AREA)
    
    def process_video(self, item):
        item_id = item.split('/')[-1].split('.')[0]
        os.makedirs("/iris/u/nivsiyer/something_something/" + item_id)
        try: 
            reader = av.open(item)
        except:
            print("Issue with opening the video, path:", item)
            assert(False)
        vidcap = cv2.VideoCapture(item)
        success,image = vidcap.read()
        image = self.resize(image, 64)
        image = self.crop(image, 64, 64)
        count = 0
        while success:
            cv2.imwrite("/iris/u/nivsiyer/something_something/%s/%s.jpg" % (str(item_id), count), image)     # save frame as JPEG file      
            success,image = vidcap.read()
            if success:
                image = self.resize(image, 64)
                image = self.crop(image, 64, 64)
            # print(image)
            print('Read a new frame: ', success)
            count += 1
    
    def dump_vids(self):
        for item in glob.glob(f'{self.root}*'):
            self.process_video(item)
        
def main():   
    dvd_data = VideoFolder(root='/iris/u/asc8/workspace/humans/Humans/20bn-something-something-v2-all-videos/')
    dvd_data.dump_vids()
   
if __name__ == '__main__':
    main()
    
