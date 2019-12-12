"""
Creating subset of full dataset
"""

import os, shutil
import glob
import json
import argparse
import numpy as np
from itertools import chain
from helper import read_block, modify_block, write_block

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Create subset of full dataset")
    parser.add_argument('--root', help='Path of the consistency repo', default='/home/arpan/BTP/consistency')
    parser.add_argument('--tag_prefix', help='Prefix to prepend before video ID in the TAG files. Default: \'\'', 
                        default='')
    parser.add_argument('--no_task', help='Specify to exclude task ID from the TAG files', action='store_true',
                        default=False)

    args = parser.parse_args()

    return args


def load_dataset (json_file):
    """
    Load the dataset, and create video_id to split mapping
    """

    with open(json_file) as json_file:
        COIN = json.load(json_file)['database']

    vid_split_map = {}
    for vid_id, annotation in COIN.items():
        vid_split_map[vid_id] = annotation['subset']

    return COIN, vid_split_map


def create_subset_folders (absolute_consistency_path):
    """
    Create the subset video and raw frame folders
    Populate using soft-links to the original files
    """
    # Creating mapping of subset of videos
    task_mapping = {}; task_no = 0
    f = open("subset",'r')
    for line in f:
        task_id = int(line.strip())
        task_mapping[task_id] = task_no
        task_no += 1

    video_path = os.path.join(absolute_consistency_path, 'data/coin/videos')
    subset_path = os.path.join(absolute_consistency_path, 'data/coin/subset')

    frames_path = os.path.join(absolute_consistency_path, 'data/coin/raw_frames')
    subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')

    # Creating symbolic links for the respective videos
    if os.path.exists(subset_path):
        shutil.rmtree(subset_path) # Remove the existing soft links
    os.mkdir(subset_path)

    for task_id in task_mapping:
        src = os.path.join(video_path, str(task_id))
        dst = os.path.join(subset_path, str(task_mapping[task_id]))
        os.symlink(src, dst)

    # Creating symbolic links for respective extracted frames
    if os.path.exists(subset_frames_path):
        shutil.rmtree(subset_frames_path) # Remove the existing soft links
    os.mkdir(subset_frames_path)

    task_folders = glob.glob(os.path.join(frames_path, '*'))
    for tf in task_folders:
        task_id = int(tf.split('/')[-1])
        if task_id in task_mapping:
            frame_folders = glob.glob(os.path.join(tf, '*'))
            for f in frame_folders:
                f_id = f.split('/')[-1]
                src = f
                dst = os.path.join(subset_frames_path, f_id)
                os.symlink(src, dst)

    return task_mapping


def create_subset_tag_files (absolute_consistency_path, prefix, vid_split_map, COIN, task_mapping, no_task=False):
    """
    Create subset TAG proposal file using the full proposal files
    """
    subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')
    coin_path = os.path.join(absolute_consistency_path, 'data/coin/')
    train_full_tag = os.path.join(coin_path, 'coin_full_tag_train_proposal_list.txt')
    test_full_tag = os.path.join(coin_path, 'coin_full_tag_test_proposal_list.txt')

    train_tag = os.path.join(coin_path, 'coin_tag_train_proposal_list.txt')
    test_tag = os.path.join(coin_path, 'coin_tag_test_proposal_list.txt')
    val_tag = os.path.join(coin_path, 'coin_tag_val_proposal_list.txt')

    vid_ids = [
        id.split('/')[-1] 
        for id in glob.glob(os.path.join(subset_frames_path, '*'))
    ]

    # Initializations
    step_mapping = {"0": "0"}
    for outfile in [train_tag, test_tag, val_tag]:
        with open(outfile, 'w') as f:
            f.write('')

    train_idx = test_idx = val_idx = 1
    blocks = chain(read_block(train_full_tag), read_block(test_full_tag))
    for block in blocks:
        id = block['id']
        split = vid_split_map[id]
        if id in vid_ids:
            task = task_mapping(str(COIN[id]['recipe_type']))
            modify_block(block, subset_frames_path, prefix, step_mapping, task)
            if split == 'training':
                write_block (block, train_tag, train_idx, no_task=no_task)
                train_idx += 1
            elif split == 'testing':
                write_block (block, test_tag, test_idx, no_task=no_task)
                test_idx += 1
            else:
                write_block (block, val_tag, val_idx, no_task=no_task)
                val_idx += 1

    return step_mapping


def create_subset_json_file (absolute_consistency_path, COIN, task_mapping, step_mapping):
    """
    Create the subset JSON dataset, using the derived task and step mappings
    """
    subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')

    frame_folders = glob.glob(os.path.join(subset_frames_path, '*'))
    video_ids = set()
    for f in frame_folders:
        vid_id = f.split('/')[-1]
        video_ids.add(vid_id)

    coin_small = {}
    for vid_id, annotation in COIN.items():
        task_id = annotation['recipe_type']
        if task_id in task_mapping and vid_id in video_ids:
            annotation['recipe_type'] = task_mapping[task_id]
            for a in annotation['annotation']:
                a["id"] = step_mapping[str(int(a["id"]) + 1)]

            coin_small[vid_id] = annotation

    return {'database': coin_small}


def create_W_matrix(COIN):
    """
    Create the belongs to matrix W
    (i, j)th entry is 1 if step i belongs to task j, otherwise 0
    """

    step_to_task_map = dict()
    tasks = set(); steps = set()
    for vid_id, annotation in COIN.items():
        task_id = annotation['recipe_type']
        if task_id not in tasks:
            tasks.add(task_id)

        for a in annotation['annotation']:
            step_id = a["id"]
            if step_id not in steps:
                steps.add(step_id)
            if step_id in step_to_task_map:
                assert step_to_task_map[step_id] == task_id
            else:
                step_to_task_map[step_id] = task_id

    W = np.zeros((len(steps), len(tasks)), dtype=int)
    for s, t in step_to_task_map.items():
        W[int(s) - 1][int(t)] = 1
        
    return step_to_task_map, W


if __name__ == '__main__':
    args = parse_args()
    absolute_consistency_path = args.root

    # Create subset folders
    task_mapping = create_subset_folders (absolute_consistency_path)
    task_mapping_path = os.path.join(absolute_consistency_path, 'data/coin/task_mapping.json')
    with open(task_mapping_path, 'w') as outfile:
        json.dump(task_mapping, outfile, indent=4)
    print ("Created subset folders.")

    # Load the dataset
    json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN_full_split.json')
    COIN, vid_split_map = load_dataset (json_path)

    # Creating subset proposal mapping
    prefix = '' if (args.tag_prefix == '') \
                else os.path.join(absolute_consistency_path, args.tag_prefix)
    step_mapping = create_subset_tag_files (absolute_consistency_path, prefix, vid_split_map, COIN, task_mapping, no_task=args.no_task)
    step_mapping_path = os.path.join(absolute_consistency_path, 'data/coin/step_mapping.json')
    with open(step_mapping_path, 'w') as outfile:
        json.dump(step_mapping, outfile, indent=4)
    print ("Created subset TAG files.")

    # Creating small JSON file
    coin_small = create_subset_json_file (absolute_consistency_path, COIN, task_mapping, step_mapping)
    coin_json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN.json')
    with open(coin_json_path, 'w') as outfile:
        json.dump(coin_small, outfile, indent=4)
    print ("Created subset JSON file")

    # Creating belonging matrix
    step_to_task_map, step_task_matrix = create_W_matrix (coin_small['database'])
    step_to_task_map_path = os.path.join(absolute_consistency_path, 'data/coin/step_to_task_map.json')
    W_matrix_file = os.path.join(absolute_consistency_path, 'data/coin/W.npy')
    with open(step_to_task_map_path, 'w') as outfile:
        json.dump(step_to_task_map, outfile, indent=4)
    np.save(W_matrix_file, step_task_matrix)
    print ("Created W matrix")

