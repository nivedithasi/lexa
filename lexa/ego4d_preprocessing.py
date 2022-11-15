path = "/nlp/scr2/nlp/ego4d/data/raw/ego4d/v1/annotations/moments_train.json"
# path = "/shared/group/ego4d/v1/annotations/moments_train.json"


import json
import pprint as pp
import os
import av
import cv2
import torch.nn as nn
from torchvision import transforms
import torchvision
import pandas as pd
import os.path
import time

from multiprocessing import Pool





f = open(path)
# returns JSON object as 
# a dictionary
data = json.load(f)

ego_path = "/iris/u/nivsiyer/ego4d/"
# ego_path = "/home/ademi_adeniji/ego4d/"

resize_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.CenterCrop(224),
    transforms.ToTensor()]
)

def process_vid(video):
    t0 = time.time()
    video_name = "/nlp/scr2/nlp/ego4d/data/raw/ego4d/v1/full_scale/" + video['video_uid'] + '.mp4'
#     video_name = "/shared/group/ego4d/v1/full_scale/" + video['video_uid'] + '.mp4'

    try: 
        reader = av.open(video_name)
    except:
        print("Issue with opening the video path:", video_name)
        return {}

    vidpath = ego_path + "videos2" + "/" + video['video_uid']

    clips = video['clips']
    manifest = {}
    manifest["label"] = []
    manifest["vidpath"] = []
    manifest["start_frame"] = []
    manifest["end_frame"] = []
    
    allvidframes = []
    numclips = len(clips)
    for e, clip in enumerate(clips):
        annotations = clip['annotations']
        for annot in annotations:
            labels = annot['labels']
            for label in labels:
                start_time = label['video_start_time']
                end_time = label['video_end_time']
                start_frame = int(label['video_start_frame'])
                end_frame = int(label['video_end_frame'])
                allvidframes += list(range(start_frame, end_frame+1))
                task_name = label['label']
                try:
                  manifest["label"].append(task_name)
                  manifest["vidpath"].append(vidpath)
                  manifest["start_frame"].append(str(start_frame).zfill(7))
                  manifest["end_frame"].append(str(end_frame).zfill(7))
                except:
                  pass
           
    if os.path.isdir(vidpath):
      print(f"Video {video['video_uid']} already Done, Process time {time.time() - t0}")
      return manifest
    os.makedirs(vidpath, exist_ok=True)
                
    allvidframes = set(allvidframes)
    for idx, frame in enumerate(reader.decode(video=0)):
        if os.path.isfile(vidpath + "/" + f"{str(idx).zfill(7)}.jpg") or (idx not in allvidframes):
          continue
        f = frame.to_image()
        f = resize_transform(f)
        torchvision.utils.save_image(f, vidpath + "/" + f"{str(idx).zfill(7)}.jpg")
    print(f"Video {video['video_uid']} Done, Process time {time.time() - t0}")
    return manifest
        

with Pool(4) as p:
  mfs = p.map(process_vid, data['videos'])

finalmanifest = {}
finalmanifest["label"] = []
finalmanifest["vidpath"] = []
finalmanifest["start_frame"] = []
finalmanifest["end_frame"] = []
for mf in mfs:
  if mf == {}:
    continue
  if len(mf["label"]) == 0:
    continue
  assert(len(mf["label"]) == len(mf["vidpath"]) == len(mf["start_frame"]) == len(mf["end_frame"]))
  
  finalmanifest["label"] += mf["label"]
  finalmanifest["vidpath"] += mf["vidpath"]
  finalmanifest["start_frame"] += mf["start_frame"]
  finalmanifest["end_frame"] += mf["end_frame"]
        
df = pd.DataFrame.from_dict(finalmanifest)
print(df)

df.to_csv("/iris/u/nivsiyer/ego4d/videos2/manifest.csv")

# df.to_csv("/home/ademi_adeniji/ego4d/videos2/manifest.csv")

