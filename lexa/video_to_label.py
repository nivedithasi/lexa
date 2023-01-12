# Referred to https://www.tensorflow.org/tutorials/video/video_classification for residual block integration

import tqdm
import random
import pathlib
import itertools
import collections

import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras as tfk

import pandas as pd
import os

from PIL import Image

# path to manifest.csv, containing video paths and class labels
videos_csv_path = "/iris/u/nivsiyer/ego4d/videos2/manifest.csv"

# name of columns to read from manifest.csv
LABEL_COL = "label"
VIDPATH_COL = "vidpath"

# dimensions of one frame in the set of frames created
HEIGHT = 64
WIDTH = 64

# number of frammes to read from video
FN = 2 

# batch size
BS = 32


class FrameGenerator:
  """
  A generator that yields a number of frames for a video and its class label.
  """
  def __init__(self, n_frames, training = False):
    self.csv_path = videos_csv_path
    self.csv = pd.read_csv(self.csv_path)
    self.n_frames = n_frames
    self.training = training
    self.class_names = sorted(set(self.csv[LABEL_COL]))
    self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

  def get_files_and_class_names(self):
    return self.csv["vidpath"], self.csv[LABEL_COL]

  def jpg_to_tensor(self, path):
    img = tf.io.read_file(path)
    tensor = np.array(tf.io.decode_image(img, channels=3, dtype=tf.dtypes.float32))
    return tensor
        

  def frames_from_video_file(self, video_dir, n_frames):
    images = os.listdir(video_dir)
    if n_frames == 2:
        # get first and last frame
        q = np.array([self.jpg_to_tensor(os.path.join(video_dir, images[0])), self.jpg_to_tensor(os.path.join(video_dir, images[-1]))])
    else:
        # read n frames from start
        # TODO: should change to sample from middle
        q = np.array([self.jpg_to_tensor(os.path.join(video_dir, images[i])) for i in range(n_frames)])
    return q

  def __call__(self):
    video_paths, classes = self.get_files_and_class_names()

    pairs = list(zip(video_paths, classes))

    if self.training:
      random.shuffle(pairs)

    for path, name in pairs:
      video_frames = self.frames_from_video_file(path, self.n_frames) 
      label = self.class_ids_for_name[name] # Encode labels
      yield video_frames, label


class Conv2Plus1D(tfk.layers.Layer):
  def __init__(self, filters, kernel_size, padding):
    """
      A sequence of convolutional layers that first apply the convolution operation over the
      spatial dimensions, and then the temporal dimension. 
    """
    super().__init__()
    self.seq = tfk.Sequential([  
        # Spatial decomposition
        layers.Conv3D(filters=filters,
                      kernel_size=(1, kernel_size[1], kernel_size[2]),
                      padding=padding),
        # Temporal decomposition
        layers.Conv3D(filters=filters, 
                      kernel_size=(kernel_size[0], 1, 1),
                      padding=padding)
        ])

  def call(self, x):
    return self.seq(x)

class ResidualMain(tfk.layers.Layer):
  """
    Residual block of the model with convolution, layer normalization, and the
    activation function, ReLU.
  """
  def __init__(self, filters, kernel_size):
    super().__init__()
    self.seq = tfk.Sequential([
        Conv2Plus1D(filters=filters,
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization(),
        layers.ReLU(),
        Conv2Plus1D(filters=filters, 
                    kernel_size=kernel_size,
                    padding='same'),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

class Project(tfk.layers.Layer):
  def __init__(self, units):
    super().__init__()
    self.seq = tfk.Sequential([
        layers.Dense(units),
        layers.LayerNormalization()
    ])

  def call(self, x):
    return self.seq(x)

def add_residual_block(input, filters, kernel_size):
  """
    Add residual blocks to the model. If the last dimensions of the input data
    and filter size does not match, project it such that last dimension matches.
  """
  out = ResidualMain(filters, 
                     kernel_size)(input)

  res = input
  # Using the Keras functional APIs, project the last dimension of the tensor to
  # match the new filter size
  if out.shape[-1] != input.shape[-1]:
    res = Project(out.shape[-1])(res)

  return layers.add([res, out])


# Model definition
input_shape = (None,FN, HEIGHT, WIDTH, 3)
input = layers.Input(shape=(input_shape[1:]))
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

x = add_residual_block(x, 16, (3, 3, 3))
x = add_residual_block(x, 32, (3, 3, 3))
x = add_residual_block(x, 64, (3, 3, 3))
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
print(x)
x = layers.Flatten()(x)
print(x)
x = layers.Dense(len(fg.class_ids_for_name))(x)

model = tfk.Model(input, x)
print(model.summary())


# Dataset definition
output_signature = (tf.TensorSpec(shape = input_shape[1:], dtype = tf.float32), tf.TensorSpec(shape = (), dtype = tf.int16))

fg = FrameGenerator(FN, training=True)
train_ds = tf.data.Dataset.from_generator(fg, output_signature= output_signature)
train_ds2 = train_ds.batch(batch_size=BS)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(FN), output_signature= output_signature)
val_ds2 = val_ds.batch(batch_size=32)

# Test how frames and labels are being sampled
frames, label = next(iter(train_ds))

# print(frames.shape)
# print(frames)
# print(label)

model.build(frames)

model.compile(loss = tfk.losses.SparseCategoricalCrossentropy(from_logits=True), 
              optimizer = tfk.optimizers.Adam(learning_rate = 0.0001), 
              metrics = ['accuracy'])

history = model.fit(x = train_ds2,
                    epochs = 50, 
                    validation_data = val_ds2,verbose=1)