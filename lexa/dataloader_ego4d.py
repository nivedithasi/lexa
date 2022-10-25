import av
import torch
import numpy as np
import os
import h5py
import pandas as pd

import os

import io
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

from dataloader_parsed import Augmentor
from tensorflow.keras.layers import Permute 
from collections import namedtuple, defaultdict, Counter
import json

ListData = namedtuple('ListData', ['label', 'vidpath', 'start_frame', 'end_frame'])
FRAMERATE = 12  # default value 

class Ego4DVideoFolder():

    def __init__(self, root, manifest_csv, clip_size=2, step_size=0, is_val=False,
                 transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 is_test=False, robot_demo_transform=None): # add back args later
        
        vidpath = "/iris/u/nivsiyer/ego4d/videos"
        self.csv = pd.read_csv(manifest_csv)
        self.labels = self.csv['label'].unique()
        self.num_tasks = len(self.labels)
        
        self.is_val = is_val

        self.root = root
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.im_size = 120 #default
        self.batch_size = 24 #args.batch_size

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.traj_length = clip_size
        self.step_size = step_size
        self.similarity = True #args.similarity
        self.add_demos = 60 #args.add_demos 
        
        classes = []
        self.classes = classes
        num_occur = defaultdict(int)
                            
        print("Number of human videos: ", len(self.csv), "Total:", self.__len__())

            
    def process_video(self, item):
         # Open video file
        try: 
            reader = av.open(item.path)
        except:
            print("Issue with opening the video, path:", item.path)
            assert(False)

        try:
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(exception).__name__, item.path))
        orig_imgs = np.array(imgs).copy() 

        p = np.array(imgs)
        imgs = self.transform_pre(imgs)
        imgs, label = self.augmentor(imgs, item.label)
        imgs = self.transform_post(imgs)

        num_frames = len(imgs)        
        if self.nclips > -1:
            num_frames_necessary = self.traj_length * self.nclips * self.step_size
        else:
            num_frames_necessary = num_frames
        offset = 0
        if num_frames_necessary < num_frames:
            # If there are more frames, then sample starting offset.
            diff = (num_frames - num_frames_necessary)
            # temporal augmentation
            offset = np.random.randint(0, diff)

        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]
        if len(imgs) < (self.traj_length * self.nclips):
            imgs.extend([imgs[-1]] *
                        ((self.traj_length * self.nclips) - len(imgs)))

        data = tf.stack(imgs)
        data = tf.transpose(tf.convert_to_tensor(data), perm=[0, 2, 3, 1])
        return data
    
            
    def __getitem__(self, indices = None):
        
        def getindices():
          anchor = random.choice(self.labels) 
          positives = self.csv[self.csv['label'] == anchor].sample(n = 2)
          neg = self.csv[self.csv['label'] != anchor].sample(n=1)
          return positives.iloc[0], positives.iloc[1], neg.iloc[0]
        
        def process(video):
          vidpath = video.vidpath
          ims = []
          # print("Version:", tf.__version__)
          load1 = tf.keras.preprocessing.image.load_img(
                  vidpath+f"/{video.start_frame:07}.jpg",
                  grayscale=False,
                  color_mode='rgb',
              )

          ims.append(load1)
          load2 = tf.keras.preprocessing.image.load_img(
                   vidpath+f"/{video.end_frame:07}.jpg",
                  grayscale=False,
                  color_mode='rgb',
              )
          ims.append(load2)
          
          ims = tf.cast(tf.stack([tf.keras.preprocessing.image.img_to_array(x) for x in ims], 0), tf.float32) / 255.0
          return ims

              
        item, anchor, neg = getindices()
        
        pos_data = process(item)
        anchor_data  =  process(anchor)
        neg_data =  process(neg)
        return tf.stack([pos_data, anchor_data, neg_data], 0) #){'pos': pos_data, 'anchor': anchor_data, 'neg': neg_data}

    def __len__(self):
        return len([name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))])
    
    def __call__(self):
        while True:
            yield 0.0
