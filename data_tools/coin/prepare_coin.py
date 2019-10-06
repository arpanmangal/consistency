import os, shutil
import glob
import json

absolute_consistency_path = '/home/arpan/BTP/consistency' # Change this path for your machine

json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN_full.json')
coin_json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN.json')
mapping_json_path = os.path.join(absolute_consistency_path, 'data/coin/mapping.json')

video_path = os.path.join(absolute_consistency_path, 'data/coin/videos')
subset_path = os.path.join(absolute_consistency_path, 'data/coin/subset')

frames_path = os.path.join(absolute_consistency_path, 'data/coin/raw_frames')
subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')

# Creating mapping of subset of videos
tasks = {}; task_no = 0
f = open("subset",'r')
for line in f:
    task_id = int(line.strip())
    tasks[task_id] = task_no
    task_no += 1

with open(mapping_json_path, 'w') as outfile:
    json.dump(tasks, outfile, indent=4)

# Creating symbolic links for the respective videos
if os.path.exists(subset_path):
    shutil.rmtree(subset_path) # Remove the existing soft links
os.mkdir(subset_path)

for task_id in tasks:
    src = os.path.join(video_path, str(task_id))
    dst = os.path.join(subset_path, str(tasks[task_id]))
    os.symlink(src, dst)

# Creating symbolic links for respective extracted frames
if os.path.exists(subset_frames_path):
    shutil.rmtree(subset_frames_path) # Remove the existing soft links
os.mkdir(subset_frames_path)

task_folders = glob.glob(os.path.join(frames_path, '*'))
for tf in task_folders:
    task_id = int(tf.split('/')[-1])
    if task_id in tasks:
        frame_folders = glob.glob(os.path.join(tf, '*'))
        for f in frame_folders:
            f_id = f.split('/')[-1]
            src = f
            dst = os.path.join(subset_frames_path, f_id)
            os.symlink(src, dst)

# Creating subset JSON file
with open(json_path) as json_file:
    COIN = json.load(json_file)['database']
coin_small = {}

frame_folders = glob.glob(os.path.join(subset_frames_path, '*'))
video_ids = []
for f in frame_folders:
    vid_id = f.split('/')[-1]
    video_ids.append(vid_id)

for vid_id, annotation in COIN.items():
    task_id = annotation['recipe_type']
    if task_id in tasks and vid_id in video_ids:
        annotation['recipe_type'] = tasks[task_id]
        coin_small[vid_id] = annotation

coin_small = {'database': coin_small}

with open(coin_json_path, 'w') as outfile:
    json.dump(coin_small, outfile, indent=4)

