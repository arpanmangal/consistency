"""
Creating the subset JSON using task and step mapping
"""

import os
import glob
import json

absolute_consistency_path = '/home/cse/btech/cs1160321/scratch/BTP/consistency' # Change this path for your machine
json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN_full.json')
coin_json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN.json')
task_mapping_json_path = os.path.join(absolute_consistency_path, 'data/coin/mapping.json')
step_mapping_json_path = os.path.join(absolute_consistency_path, 'data/coin/proposal_mapping.json')

subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')


with open(task_mapping_json_path, 'r') as f:
    task_map = json.load(f)

with open(step_mapping_json_path, 'r') as f:
    step_map = json.load(f)

with open(json_path) as json_file:
    COIN = json.load(json_file)['database']
coin_small = {}


frame_folders = glob.glob(os.path.join(subset_frames_path, '*'))
video_ids = []
for f in frame_folders:
    vid_id = f.split('/')[-1]
    video_ids.append(vid_id)

for vid_id, annotation in COIN.items():
    task_id = str(annotation['recipe_type'])
    if task_id in task_map and vid_id in video_ids:
        annotation['recipe_type'] = task_map[task_id]
        annotations = annotation['annotation']
        for a in annotations:
            a["id"] = step_map[str(int(a["id"]) + 1)]

        coin_small[vid_id] = annotation

coin_small = {'database': coin_small}

with open(coin_json_path, 'w') as outfile:
    json.dump(coin_small, outfile, indent=4)
