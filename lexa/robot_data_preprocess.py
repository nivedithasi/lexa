import imageio
import os
import numpy as np
import shutil
import envs
import pandas as pd

# create no-object interact videos
if os.path.exists("gifs_class2"):
    shutil.rmtree("gifs_class2")
os.makedirs("gifs_class2")

vid_num = 1
env = envs.RoboBinEnv(3, False, False)
while vid_num < 61:
    obs = env.reset()
    video = [obs['image']]
    for i in range(49):
        obs, total_reward, done, info = env.step(env.action_space.sample())
        video.append(obs['image'])
        env.close()
    print('saving video')
    imageio.mimsave(f'gifs_class2/videos{vid_num}.gif', video, duration=0.1)
    vid_num += 1

if os.path.exists("robot_videos"):
    shutil.rmtree("robot_videos")
os.makedirs("robot_videos")

def preprocess(object_interact, manifest, idx):
    if object_interact:
        # object interact videos should be saved in gifs_class1
        gifs_path = "gifs_class1/"
        gifs = []

        if os.path.exists("gifs_64x64"):
            shutil.rmtree("gifs_64x64")
        os.makedirs("gifs_64x64")
    else:
        gifs_path = "gifs_class2/"
        gifs = []

    for gif in os.listdir(gifs_path):
        gif_array = np.array(imageio.mimread(gifs_path + gif))
        if object_interact:
            gif_segments = [gif_array[:, 0:64, k:k+64, :] for k in range(0, 384, 64)]
        else:
            gif_segments = [gif_array]
        gifs.extend(gif_segments)

    print(f'Loaded {len(gifs)} videos')
        
    for i in range(len(gifs)):
        if object_interact:
            imageio.mimsave(f"gifs_64x64/gif{i}.gif", gifs[i])
        os.makedirs(f"robot_videos/{idx}")
        for j in range(gifs[i].shape[0]):
            # get rid of alpha channel from gifs
            imageio.imwrite(f"robot_videos/{i}/{str(j).zfill(7)}.jpg", gifs[i][j][:, :, :3])
    
        manifest['label'].append('object_interact' if object_interact else 'no_object_interact')
        manifest['start_frame'].append(str(0).zfill(7))
        manifest['end_frame'].append(str(gifs[i].shape[0]-1).zfill(7))
        manifest['vidpath'].append(f"/home/ademi_adeniji/lexastuff/lexa_dvd/lexa/robot_videos/{idx}")
        idx += 1
    return idx, manifest

manifest = {}
manifest["label"] = []
manifest["vidpath"] = []
manifest["start_frame"] = []
manifest["end_frame"] = []
idx = 0
idx, manifest = preprocess(True, manifest, idx)
idx, manifest = preprocess(False, manifest, idx)
df = pd.DataFrame.from_dict(manifest)
print(df)
df.to_csv('robot_videos/manifest.csv')
