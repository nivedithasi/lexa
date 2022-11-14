import av
import torch
import numpy as np
import os
import h5py
import pandas as pd

import os
import os.path

import io
import json
import numpy as np
# import torchvision
# from transforms_video import *
# from tensorflow import *
# import torchvision
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
    
    def frames_exist(self, video):
            vidpath = video.vidpath
            start_framepath = vidpath+f"/{video.start_frame:07}.jpg"
            end_framepath = vidpath+f"/{video.end_frame:07}.jpg"
            return os.path.isfile(start_framepath) and os.path.isfile(end_framepath) 
        

    def getindices(self):
          anchor = random.choice(self.labels)
          pos = self.csv[self.csv['label'] == anchor]
          bothdone = False
            
          
          if len(pos) >= 2:
            pos2 = pos.sample(n=2)
            positives1 = pos2.iloc[0]
            positives2 = pos2.iloc[1]
            bothdone = self.frames_exist(positives1) and self.frames_exist(positives2) and positives1['vidpath'] != positives2['vidpath']

          while not bothdone:
                anchor = random.choice(self.labels)
                pos = self.csv[self.csv['label'] == anchor]
                if len(pos) >= 2:
                    pos2 = pos.sample(n=2)
                    positives1 = pos2.iloc[0]
                    positives2 = pos2.iloc[1]
                    bothdone = self.frames_exist(positives1) and self.frames_exist(positives2) and positives1['vidpath'] != positives2['vidpath']
          
          neg = self.csv[self.csv['label'] != anchor].sample(n=1)
          while not self.frames_exist(neg.iloc[0]):
            neg = self.csv[self.csv['label'] != anchor].sample(n=1)
            
          return positives1, positives2, neg.iloc[0], anchor, neg.iloc[0]['label']
    
    
    def process(self, video):
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
    

    def get_ims_labels(self):
        item, anchor, neg, pos_anchor, neg_anchor = self.getindices()

        pos_data = self.process(item)
        anchor_data  =  self.process(anchor)
        neg_data =  self.process(neg)
        
        return pos_data, anchor_data, neg_data, anchor, pos_anchor, neg_anchor
        
        
    def __getitem__(self, indices = None, get_labels=False):
                            
        item, anchor, neg, pos_anchor, neg_anchor = self.getindices()

        pos_data = self.process(item)
        anchor_data  = self.process(anchor)
        neg_data =  self.process(neg)
        return tf.stack([pos_data, anchor_data, neg_data], 0)

    def __len__(self):
        return len([name for name in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, name))])
    
    def __call__(self):
        while True:
            yield 0.0
