import glob
import os
import json

base_path = '../../data/coin/'
dataset_path = os.path.join(base_path, 'raw_rgb_frames')
coin_json_path = os.path.join(base_path, 'COIN.json')
small_coin_json_path = os.path.join(base_path, 'COIN_small.json')

with open(coin_json_path) as json_file:
    COIN = json.load(json_file)['database']

coin_small = {}
task_folders = glob.glob(os.path.join(dataset_path, '*'))
for tf in task_folders:
    frame_folders = glob.glob(os.path.join(tf, '*'))
    for f in frame_folders:
        f_id = f.split('/')[-1]
        coin_small[f_id] = COIN[f_id]

coin_small = {'dataset': coin_small}

with open(small_coin_json_path, 'w') as outfile:
    json.dump(coin_small, outfile, indent=4)