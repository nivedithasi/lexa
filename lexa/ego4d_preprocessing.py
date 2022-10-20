path = "/nlp/scr2/nlp/ego4d/data/raw/ego4d/v1/annotations/moments_train.json"

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

f = open(path)
  
# returns JSON object as 
# a dictionary
data = json.load(f)


data_logger = []
ego_path = "/iris/u/nivsiyer/ego4d/"

resize_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.CenterCrop(224),
    transforms.ToTensor()]
)

numvids = len(data['videos'])
for i, video in enumerate(data['videos']):
    print("*" * 50)
    print("Video", video['video_uid'], f"{i} of {numvids}")
    
    video_name = "/nlp/scr2/nlp/ego4d/data/raw/ego4d/v1/full_scale/" + video['video_uid'] + '.mp4'
    try: 
        reader = av.open(video_name)
    except:
        print("Issue with opening the video path:", video_name)
        assert(False)
        
    # Convert video into a list of images
    # images = []
    
    vidpath = ego_path + "videos" + "/" + video['video_uid']
    os.makedirs(vidpath, exist_ok=True)
#     for idx, frame in enumerate(reader.decode(video=0)):
#         if os.path.isfile(vidpath + "/" + f"{str(idx).zfill(7)}.jpg"):
#           continue
        
#         f = frame.to_image()
#         f = resize_transform(f)
#         # images.append(f)
#         torchvision.utils.save_image(f, vidpath + "/" + f"{str(idx).zfill(7)}.jpg")

    clips = video['clips']
    # print(clips)
    manifest = {}
    manifest["label"] = []
    manifest["vidpath"] = []
    manifest["start_frame"] = []
    manifest["end_frame"] = []
    
    allvidframes = []
    
    numclips = len(clips)
    for e, clip in enumerate(clips):
        print(f"Clip {e} of {numclips}")
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
                print(task_name, start_frame, end_frame)
                try:
                  manifest["label"].append(task_name)
                  manifest["vidpath"].append(vidpath)
                  manifest["start_frame"].append(str(start_frame).zfill(7))
                  manifest["end_frame"].append(str(end_frame).zfill(7))
                except:
                  pass
                
    allvidframes = set(allvidframes)
    
    for idx, frame in enumerate(reader.decode(video=0)):
        if os.path.isfile(vidpath + "/" + f"{str(idx).zfill(7)}.jpg") or (idx not in allvidframes):
          continue
        
        f = frame.to_image()
        f = resize_transform(f)
        torchvision.utils.save_image(f, vidpath + "/" + f"{str(idx).zfill(7)}.jpg")
    
df = pd.DataFrame.from_dict(manifest)
print(df)
df.to_csv("/iris/u/nivsiyer/ego4d/videos/manifest.csv")