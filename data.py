import numpy as np
import pickle
import os
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from data_s2s import VideoDataset
import torch.utils.data as data
import pdb

def set_up_data_s2s(H):
    train_dataset = VideoDataset('/shared/ademi_adeniji/something-something/drawers', H.sequence_length,
                        train=True, resolution=H.resolution, frame_mode=H.frame_mode)
    val_dataset = VideoDataset('/shared/ademi_adeniji/something-something/drawers', H.sequence_length,
                        train=False, resolution=H.resolution, frame_mode=H.frame_mode)
    train_sampler = data.DistributedSampler(train_dataset, num_replicas=H.mpi_size, rank=H.rank)
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=H.n_batch // H.mpi_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
    )
    val_sampler = data.DistributedSampler(val_dataset, num_replicas=H.mpi_size, rank=H.rank)
    val_dataloader = data.DataLoader(
        val_dataset,
        batch_size=H.n_batch // H.mpi_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        sampler=val_sampler,
    )
    viz_sampler = data.DistributedSampler(val_dataset, num_replicas=H.mpi_size, rank=H.rank)
    viz_dataloader = data.DataLoader(
        val_dataset,
        batch_size=H.num_images_visualize // H.mpi_size,
        num_workers=4,
        pin_memory=True,
        sampler=viz_sampler,
    )
    H.image_size = H.resolution
    H.image_channels = 3
    return H, train_dataloader, val_dataloader, viz_dataloader