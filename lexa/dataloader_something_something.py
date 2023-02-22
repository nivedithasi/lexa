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
from transforms_video import *
import tensorflow as tf
import pdb
import time
from dataloader_parsed import Augmentor
from tensorflow.keras.layers import Permute 
from collections import namedtuple, defaultdict, Counter
import json

from sklearn import preprocessing

ListData = namedtuple('ListData', ['id', 'label', 'template', 'placeholders', 'template_cleaned', 'numeric_label'])
FRAMERATE = 12  # default value 

class SthSthFolder():

    def __init__(self, root, manifest_csv, clip_size=2, step_size=0, is_val=False,
                 transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 is_test=False, robot_demo_transform=None, classifier=False): # add back args later
        
        vidpath = "/iris/u/nivsiyer/something_something"
        self.vidpath = vidpath
        self.csv = pd.read_csv(manifest_csv)
        self.labels = self.csv['numeric_label'].unique()
        self.num_tasks = len(self.labels)
        self.classifier = classifier
        
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
        
        self.le = preprocessing.LabelBinarizer()
        self.le.fit(self.labels)
        self.categories = len(self.le.classes_)

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
    
    def getindices(self):
          anchor = random.choice(self.labels)
          pos = self.csv[self.csv['numeric_label'] == anchor]
          bothdone = False
            
          
          if len(pos) >= 2:
            pos2 = pos.sample(n=2)
            positives1 = pos2.iloc[0]
            positives2 = pos2.iloc[1]
#             bothdone = self.frames_exist(positives1) and self.frames_exist(positives2) #and (positives1.vidpath != positives2.vidpath)

#           while not bothdone:
#             anchor = random.choice(self.labels)
#             pos = self.csv[self.csv['label'] == anchor]
#             if len(pos) >= 2:
#                 pos2 = pos.sample(n=2)
#                 positives1 = pos2.iloc[0]
#                 positives2 = pos2.iloc[1]
#                 bothdone = self.frames_exist(positives1) and self.frames_exist(positives2) #and (positives1.vidpath != positives2.vidpath)
          
          neg = self.csv[self.csv['label'] != anchor].sample(n=1)
#           while not self.frames_exist(neg.iloc[0]):
#             neg = self.csv[self.csv['label'] != anchor].sample(n=1)
            
          while True:
            glabel = random.choice(self.labels)
            # 'Pushing something from left to right': '93',
            # 'Pushing something from right to left': '94',
            # 'Picking something up': '47','
            # 'Putting something on a surface': '109',
            if glabel==93 or glabel==47 or glabel==109: # ---------------------------------> TODO: change glables for sth_sth
              break
          guide = self.csv[self.csv['numeric_label'] == glabel]
          g = guide.sample(n=1)
          guideclip = g.iloc[0]
          
            
          return positives1, positives2, neg.iloc[0], anchor, neg.iloc[0]['label'], guideclip
    
    
    def process(self, folder):
        ims = []
        alpha = 0.3
        start_frame = '0'
                
        def number_imgs(DIR):
            return len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
        
        folder = str(folder)
        num_ims = number_imgs(os.path.join(self.vidpath, folder))
        
        load1 = tf.keras.preprocessing.image.load_img(
              os.path.join(self.vidpath, folder+f"/{start_frame}.jpg"),
              grayscale=False,
              color_mode='rgb'
        )
        ims.append(load1)
        
        
        load2 = tf.keras.preprocessing.image.load_img(
            os.path.join(self.vidpath, folder+f"/{num_ims-1}.jpg"),
            grayscale=False,
            color_mode='rgb')

        ims.append(load2)
        ims = tf.cast(tf.stack([tf.keras.preprocessing.image.img_to_array(x) for x in ims], 0), tf.float32) / 255.0
        return ims
    

    def get_ims_labels(self):
        item, anchor, neg, pos_anchor, neg_anchor, guideclip = self.getindices()

        pos_data = self.process(item)
        anchor_data  =  self.process(anchor)
        neg_data =  self.process(neg)
        g_data =  self.process(guideclip)
        
        return pos_data, anchor_data, neg_data, anchor, pos_anchor, neg_anchor, g_data
        
        
    def __getitem__(self, indices = None, get_labels=False):
        if self.classifier:
          label = random.choice(self.labels)
          pos = self.csv[self.csv['numeric_label'] == label]
          pos = pos.sample(n=1).iloc[0]
          pos_data  = self.process(pos['id'])
          finallabel = tf.cast(tf.stack(self.le.transform([label]))[0], tf.float16)
#           for label_id in range(len(self.labels)):
#               print("before our labels")
#               print(self.labels[label_id], self.csv[self.csv['numeric_label'] == self.labels[label_id]].iloc[0]["template_cleaned"])
#           import sys
#           sys.exit()
          return (pos_data, finallabel)
        else:
          item, anchor, neg, pos_anchor, neg_anchor, guideclip = self.getindices()
          pos_data = self.process(item['id'])
          anchor_data  = self.process(anchor['id'])
          neg_data =  self.process(neg['id'])
          g_data =  self.process(guideclip['id'])
          return tf.stack([pos_data, anchor_data, neg_data, g_data], 0)

    def __len__(self):
        return len(self.csv)
    
    def __call__(self):
        while True:
            yield 0.0
