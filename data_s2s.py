from multiprocessing.sharedctypes import Value
import os
import os.path as osp
import math
import random
import pickle
import warnings

import glob
import numpy as np
import pdb

import torch
import torch.utils.data as data
import torch.nn.functional as F
import torch.distributed as dist
from torchvision.datasets.video_utils import VideoClips
import pytorch_lightning as pl

class MetaworldDataset(data.Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        data = np.moveaxis(self.data[:, idx, :, :, :], -1, 0)
        data_normed = -1 + 2 * data.astype(np.float32) / 255 
        return data_normed
        
class MnistDataset(data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return self.data.shape[1]

    def __getitem__(self, idx):
        data = np.repeat(np.expand_dims(self.data[:, idx, :, :], 0), 3, 0)
        data_normed = -1 + 2 * data.astype(np.float32) / 255 
        return data_normed

class VideoDataset(data.Dataset):
    """ Generic dataset for videos files stored in folders
    Returns BCTHW videos in the range [-0.5, 0.5] """
    exts = ['avi', 'mp4', 'webm']

    def __init__(self, data_folder, sequence_length, train=True, resolution=64, frame_mode=None):
        """
        Args:
            data_folder: path to the folder with videos. The folder
                should contain a 'train' and a 'test' directory,
                each with corresponding videos stored
            sequence_length: length of extracted video sequences
        """
        super().__init__()
        self.train = train
        self.sequence_length = sequence_length
        self.resolution = resolution
        self.frame_mode = frame_mode
        if self.frame_mode not in ["reconstruct", "final", "horizon", "delta"]:
            raise ValueError("frame data type incorrectly specified")

        folder = osp.join(data_folder, 'train' if train else 'test')
        files = sum([glob.glob(osp.join(folder, '**', f'*.{ext}'), recursive=True)
                     for ext in self.exts], [])

        # hacky way to compute # of classes (count # of unique parent directories)
        self.classes = list(set([get_parent_dir(f) for f in files]))
        self.classes.sort()
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}

        warnings.filterwarnings('ignore')
        cache_file = osp.join(folder, f"metadata_{sequence_length}.pkl")
        if not osp.exists(cache_file):
            clips = VideoClips(files, sequence_length, num_workers=32)
            pickle.dump(clips.metadata, open(cache_file, 'wb'))
        else:
            metadata = pickle.load(open(cache_file, 'rb'))
            clips = VideoClips(files, sequence_length,
                               _precomputed_metadata=metadata)
        self._clips = clips

    @property
    def n_classes(self):
        return len(self.classes)

    def __len__(self):
        return self._clips.num_clips()

    def __getitem__(self, idx):
        resolution = self.resolution
        video, _, _, idx = self._clips.get_clip(idx)
        video = preprocess(video, resolution)
        if self.frame_mode == "reconstruct":
            return (video[:, 0, :, :], video[:, 0, :, :], 0)
        elif self.frame_mode == "final":
            return (video[:, 0, :, :], video[:, -1, :, :], 0)
        elif self.frame_mode == "delta":
            return (video[:, 0, :, :], video[:, -1, :, :] - video[:, 0, :, :], 0)
        elif self.frame_mode == "horizon":
            start = random.randint(0, self.sequence_length-2)
            end = random.randint(start, self.sequence_length-1)
            horizon = end - start + 1
            return (video[:, start, :, :], video[:, end, :, :], horizon)
        else:
            raise ValueError("No valid frame type specified")
            
        



def get_parent_dir(path):
    return osp.basename(osp.dirname(path))


def preprocess(video, resolution, sequence_length=None):
    # video: THWC, {0, ..., 255}
    video = video.permute(0, 3, 1, 2).float() / 255. # TCHW
    t, c, h, w = video.shape

    # temporal crop
    if sequence_length is not None:
        assert sequence_length <= t
        video = video[:sequence_length]

    # scale shorter side to resolution
    scale = resolution / min(h, w)
    if h < w:
        target_size = (resolution, math.ceil(w * scale))
    else:
        target_size = (math.ceil(h * scale), resolution)
    video = F.interpolate(video, size=target_size, mode='bilinear',
                          align_corners=False)

    # center crop
    t, c, h, w = video.shape
    w_start = (w - resolution) // 2
    h_start = (h - resolution) // 2
    video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]
    video = video.permute(1, 0, 2, 3).contiguous() # CTHW

    video -= 0.5

    return video


class VideoData(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    @property
    def n_classes(self):
        dataset = self._dataset(True)
        return dataset.n_classes


    def _dataset(self, train):
        Dataset = VideoDataset 
        dataset = Dataset(self.args.data_path, self.args.sequence_length,
                          train=train, resolution=self.args.resolution)
        return dataset


    def _dataloader(self, train):
        dataset = self._dataset(train)
        if dist.is_initialized():
            sampler = data.distributed.DistributedSampler(
                dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
            )
        else:
            sampler = None
        dataloader = data.DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            sampler=sampler,
            shuffle=sampler is None
        )
        if train:
            self.train = dataset
            self.train_loader = dataloader
        else:
            self.test = dataset
            self.test_loader = dataloader
        return dataloader

    def train_dataloader(self):
        return self._dataloader(True)

    def val_dataloader(self):
        return self._dataloader(False)

    def test_dataloader(self):
        return self.val_dataloader()