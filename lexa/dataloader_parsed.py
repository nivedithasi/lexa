import av
import numpy as np
import os

import os

import io
import json
import numpy as np
from transforms_video import *
import tensorflow as tf
import time

from tensorflow.keras.layers import Permute 
from collections import namedtuple, defaultdict, Counter
import json

ListData = namedtuple('ListData', ['id', 'label', 'path'])
FRAMERATE = 12  # default value

# print(tf.__version__)
class Augmentor(object):
    def __init__(self, augmentation_mappings_json=None,
                 augmentation_types_todo=None,
                 fps_jitter_factors=[1, 0.75, 0.5]):
        self.augmentation_mappings_json = augmentation_mappings_json
        self.augmentation_types_todo = augmentation_types_todo
        self.fps_jitter_factors = fps_jitter_factors

        # read json to get the mapping dict
        self.augmentation_mapping = self.read_augmentation_mapping(
                                        self.augmentation_mappings_json)
        self.augmentation_transforms = self.define_augmentation_transforms()

    def __call__(self, imgs, label):
        if not self.augmentation_mapping:
            return imgs, label
        else:
            candidate_augmentations = {"same": label}
            for candidate in self.augmentation_types_todo:
                if candidate == "jitter_fps":
                    continue
                if label in self.augmentation_mapping[candidate]:
                    if isinstance(self.augmentation_mapping[candidate], list):
                        candidate_augmentations[candidate] = label
                    elif isinstance(self.augmentation_mapping[candidate], dict):
                        candidate_augmentations[candidate] = self.augmentation_mapping[candidate][label]
                    else:
                        print("Something wrong with data type specified in "
                              "augmentation file. Please check!")
            augmentation_chosen = np.random.choice(list(candidate_augmentations.keys()))
            imgs = self.augmentation_transforms[augmentation_chosen](imgs)
            label = candidate_augmentations[augmentation_chosen]

            return imgs, label

    def read_augmentation_mapping(self, path):
        if path:
            with open(path, "rb") as fp:
                mapping = json.load(fp)
        else:
            mapping = None
        return mapping

    def define_augmentation_transforms(self, ):
        augmentation_transforms = {}
        augmentation_transforms["same"] = IdentityTransform()
        augmentation_transforms["left/right"] = RandomHorizontalFlipVideo(1)
        augmentation_transforms["left/right agnostic"] = RandomHorizontalFlipVideo(1)
        augmentation_transforms["reverse time"] = RandomReverseTimeVideo(1)
        augmentation_transforms["reverse time agnostic"] = RandomReverseTimeVideo(0.5)

        return augmentation_transforms

    def jitter_fps(self, framerate):
        if self.augmentation_types_todo and "jitter_fps" in self.augmentation_types_todo:
            jitter_factor = np.random.choice(self.fps_jitter_factors)
            return int(jitter_factor * framerate)
        else:
            return framerate
        
        
class DatasetBase(object):
    """
    To read json data and construct a list containing video sample `ids`,
    `label` and `path`
    """
    def __init__(self, json_path_input, json_path_labels, data_root,
                 extension, num_tasks, is_test=False, is_val=False): # add args back
        self.num_tasks = num_tasks
        self.json_path_input = json_path_input
        self.json_path_labels = json_path_labels
        self.data_root = data_root
        self.extension = extension
        self.is_test = is_test
        self.is_val = is_val
        self.just_robot = False #args.just_robot
        self.sim_dir = 'demos' #args.sim_dir
        
        self.num_occur = defaultdict(int)
        
        self.tasks = [5, 41, 93] #args.human_tasks
        self.add_demos = 60 #args.add_demos
        if self.add_demos:
            self.robot_tasks = [5, 41, 93] #args.robot_tasks

        # preparing data and class dictionary
        self.classes = self.read_json_labels()
        self.classes_dict = self.get_two_way_dict(self.classes)
        self.json_data = self.read_json_input()
        print("Number of human videos:", self.num_occur.values())
        
        
    def read_json_input(self):
        json_data = []
        if not self.is_test:
            if not self.just_robot: #not self.triplet or not self.add_demos: #self.is_val or
                with open(self.json_path_input, 'rb') as jsonfile:
                    json_reader = json.load(jsonfile)
                    for elem in json_reader:
                        label = self.clean_template(elem['template'])
                        if label not in self.classes_dict.keys(): # or label == 'Pushing something so that it slightly moves':
                            continue
                        if label not in self.classes:
                            raise ValueError("Label mismatch! Please correct")
                        
                        label_num = self.classes_dict[label]
                        item = ListData(elem['id'],
                                        label,
                                        os.path.join(self.data_root,
                                                     elem['id'] + self.extension)
                                        )
                        json_data.append(item)
                        self.num_occur[label] += 1
            
            """
            FIX THIS
            """
            self.add_demos = 0
            if self.add_demos: 
                # Add robot demonstrations or extra robot class to json_data, just use id 300000
                robot_tasks = self.robot_tasks
                root_in_dir = self.sim_dir 
                for label_num in robot_tasks: 
                    # add task demos for task label_num
                    in_dirs = [f'{root_in_dir}/env1/task{label_num}_webm', f'{root_in_dir}/env1_rearranged/task{label_num}_webm']
                        
                    for in_dir in in_dirs:
                        label = self.classes_dict[label_num]

                        num_demos = self.add_demos
                        self.num_occur[label] += num_demos
                        if not self.is_val: 
                            for j in range(num_demos):
                                item = ListData(300000,
                                            label,
                                            os.path.join(in_dir, str(j) + self.extension)
                                            )
                                json_data.append(item)
                        else:
                            for j in range(num_demos, int(1.4*num_demos)):
                                item = ListData(300000,
                                            label,
                                            os.path.join(in_dir, str(j) + self.extension)
                                            )
                                json_data.append(item)
                        

        else:
            with open(self.json_path_input, 'rb') as jsonfile:
                json_reader = json.load(jsonfile)
                for elem in json_reader:
                    # add a dummy label for all test samples
                    item = ListData(elem['id'],
                                    "Holding something",
                                    os.path.join(self.data_root,
                                                 elem['id'] + self.extension)
                                    )
                    json_data.append(item)
        return json_data

    def read_json_labels(self):
        classes = []
        with open(self.json_path_labels, 'rb') as jsonfile:
            json_reader = json.load(jsonfile)
            for elem in json_reader:
                classes.append(elem)
        return sorted(classes)

    def get_two_way_dict(self, classes):
        classes_dict = {} 
        tasks = self.tasks
        for i, item in enumerate(classes):
            if i not in tasks:
                continue
            classes_dict[item] = i
            classes_dict[i] = item
        print("Length of keys", len(classes_dict.keys()), classes_dict.keys())
        return classes_dict

    def clean_template(self, template):
        """ Replaces instances of `[something]` --> `something`"""
        template = template.replace("[", "")
        template = template.replace("]", "")
        return template


class WebmDataset(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root, num_tasks, 
                 is_test=False, is_val=False): # add args back
        EXTENSION = ".webm"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, num_tasks, is_test, is_val)


class I3DFeatures(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".npy"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)


class ImageNetFeatures(DatasetBase):
    def __init__(self, json_path_input, json_path_labels, data_root,
                 is_test=False):
        EXTENSION = ".npy"
        super().__init__(json_path_input, json_path_labels, data_root,
                         EXTENSION, is_test)
        
        
class VideoFolder():

    def __init__(self, root, json_file_input, json_file_labels, clip_size,
                 nclips, step_size, is_val, num_tasks=174, transform_pre=None, transform_post=None,
                 augmentation_mappings_json=None, augmentation_types_todo=None,
                 is_test=False, robot_demo_transform=None): # add back args later
        self.num_tasks = num_tasks
        self.is_val = is_val
        self.dataset_object = WebmDataset(json_file_input, json_file_labels,
                                      root, num_tasks=self.num_tasks, is_test=is_test, is_val=is_val)
        self.json_data = self.dataset_object.json_data
        self.classes = self.dataset_object.classes
        self.classes_dict = self.dataset_object.classes_dict
        self.root = root
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.im_size = 120 #default
        self.batch_size = 24 #args.batch_size

        self.augmentor = Augmentor(augmentation_mappings_json,
                                   augmentation_types_todo)

        self.traj_length = clip_size
        self.nclips = nclips
        self.step_size = step_size
        self.similarity = True #args.similarity
        self.add_demos = 60 #args.add_demos 
        if self.add_demos:
            self.robot_demo_transform = robot_demo_transform
            self.demo_batch_val = 0.5 #args.demo_batch_val
        
        classes = []
        for key in self.classes_dict.keys():
            if not isinstance(key, int):
                classes.append(key)
        self.classes = classes
        num_occur = defaultdict(int)
        for c in self.classes:
            for video in self.json_data:
                if video.label == c:
                    num_occur[c] += 1
        if not self.is_val: #config.logdir
            with open('test_run' + '/human_data_tasks.txt', 'w') as f:
                json.dump(num_occur, f, indent=2)
        else:
            with open('test_run' + '/val_human_data_tasks.txt', 'w') as f:
                json.dump(num_occur, f, indent=2)
                
        # Every sample in batch: anchor (randomly selected class A), positive (randomly selected class A), 
        # and negative (randomly selected class not A)
        # Make dictionary for similarity triplets
        self.json_dict = defaultdict(list)
        for data in self.json_data:
            self.json_dict[data.label].append(data)

        # Make separate robot dictionary:
        self.robot_json_dict = defaultdict(list)
        #print("json data", self.json_data)
        self.total_robot = [] # all robot demos
        for data in self.json_data:
            if data.id == 300000: # robot video
                self.robot_json_dict[data.label].append(data)
                self.total_robot.append(data)
            
        print("Number of human videos: ", len(self.json_data), len(self.classes), "Total:", self.__len__())
        
        # Tasks used
        self.tasks = [5, 41, 93] #args.human_tasks
        if self.add_demos:
            self.robot_tasks = [5, 41, 93] # args.robot_tasks
        assert(sum(num_occur.values()) == len(self.json_data))        
            
    def process_video(self, item):
         # Open video file
        try: 
            reader = av.open(item.path)
        except:
            print("Issue with opening the video, path:", item.path)
            assert(False)

        # print("No issue opening")
        try:
            imgs = []
            imgs = [f.to_rgb().to_ndarray() for f in reader.decode(video=0)]
        except (RuntimeError, ZeroDivisionError) as exception:
            print('{}: WEBM reader cannot open {}. Empty '
                  'list returned.'.format(type(exception).__name__, item.path))
        orig_imgs = np.array(imgs).copy() 
#         imgs = list(imgs)
        
        # print("No issue decoding")
        target_idx = self.classes_dict[item.label] 
        if not self.num_tasks == 174:
            target_idx = self.tasks.index(target_idx)
            
        # If robot demonstration
#         if self.add_demos and item.id == 300000: 
#             imgs = self.robot_demo_transform(imgs)
#             frame = random.randint(0, max(len(imgs) - self.traj_length, 0))
#             length = min(self.traj_length, len(imgs))
#             imgs = imgs[frame: length + frame]
#             imgs_copy = torch.stack(imgs)
#             imgs_copy = imgs_copy.permute(1, 0, 2, 3)
#             return imgs_copy
        
        # print("Before trnasform pre")
        p = np.array(imgs)
        # print(p.shape)
        imgs = self.transform_pre(imgs)
        # this is a list of PIL Images
        # print("Before augment")
#         p = tf.convert_to_tensor(imgs)
#         print(p.shape)
        imgs, label = self.augmentor(imgs, item.label)
        imgs = self.transform_post(imgs)
        # print("after post augmentor")
#         print(imgs)
        
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

        # print("after nclips")
        imgs = imgs[offset: num_frames_necessary + offset: self.step_size]
        if len(imgs) < (self.traj_length * self.nclips):
            imgs.extend([imgs[-1]] *
                        ((self.traj_length * self.nclips) - len(imgs)))

        # format data to torch
        data = tf.stack(imgs)
        # print("This is data: ", type(data), data.shape)
#         data = tf.Tensor(data, data.shape, dtype=int32)
        data = tf.transpose(tf.convert_to_tensor(data), perm=[0, 2, 3, 1])
        return data
    
            
    def __getitem__(self, indices = None):
        """
        [!] FPS jittering doesn't work with AV dataloader as of now
        """
        
        def getindices():
          # for _ in range(100):
            # item = random.choice(self.json_data) 
            # print(item)
          # assert(False)
          item = random.choice(self.json_data) 
        #   while not os.path.exists(item.path):
        #     item = random.choice(self.json_data) 
        #   import pdb; pdb.set_trace()
          anchor = random.choice(self.json_dict[item.label])
          neg = random.choice(self.json_data)
          while neg.label == item.label:
              neg = random.choice(self.json_data)
          return item, anchor, neg
        
        def process(vidid):
          vidpath = f"/shared/ademi_adeniji/something-something-dvd/{vidid}"
          vidlen = len(os.listdir(vidpath))
          # print(vidlen)
          start =  np.random.randint(0, vidlen * 0.2)
          end = np.random.randint(vidlen * 0.8, vidlen)
          # start = np.random.randint(0, max(1,vidlen - self.traj_length))
          # end = min(vidlen, start+self.traj_length) - 1
          ims = []
          # print("Version:", tf.__version__)
          load1 = tf.keras.preprocessing.image.load_img(
                  vidpath+f"/{start}.jpg",
                  grayscale=False,
                  color_mode='rgb',
                  # target_size=(64,64),
                  # interpolation='bilinear',
              )
          # print(load1)
          ims.append(load1)
          load2 = tf.keras.preprocessing.image.load_img(
                  vidpath+f"/{end}.jpg",
                  grayscale=False,
                  color_mode='rgb',
                  # target_size=(64,64),
                  # interpolation='bilinear',
              )
          ims.append(load2)
          
          ims = tf.cast(tf.stack([tf.keras.preprocessing.image.img_to_array(x) for x in ims], 0), tf.float32) / 255.0
          return ims

              
        t0 = time.time()
        item, anchor, neg = getindices()
        
        pos_data = process(item.id)
        anchor_data  =  process(anchor.id)
        neg_data =  process(neg.id)
        return tf.stack([pos_data, anchor_data, neg_data], 0) #){'pos': pos_data, 'anchor': anchor_data, 'neg': neg_data}

    def __len__(self):
        self.total_files = len(self.json_data)
        if self.similarity and not self.is_val and self.num_tasks <= 12:
            self.total_files = self.batch_size * 200 
        return self.total_files
    
    def __call__(self):
        while True:
            # yield self.__getitem__()
            # item = random.choice(self.json_data) 
            # anchor = random.choice(self.json_dict[item.label])
            # neg = random.choice(self.json_data)
            # while neg.label == item.label:
            #     neg = random.choice(self.json_data)
            yield 0.0
        # for i in range(self.__len__()):
        #     item = self.__getitem__(i)
        #     yield item
            