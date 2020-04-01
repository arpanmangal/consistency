"""
Preparing the full coin dataset
"""

import os, shutil
import glob
import json
import argparse
import numpy as np
from itertools import chain
from helper import read_block, modify_block_simple, write_block

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Prepare the full dataset")
    parser.add_argument('--root', help='Path of the consistency repo', default='/home/arpan/BTP/consistency')
    parser.add_argument('--tag_prefix', help='Prefix to prepend before video ID in the TAG files. Default: \'\'', 
                        default='')
    parser.add_argument('--no_task', help='Specify to exclude task ID from the TAG files', action='store_true',
                        default=False)

    args = parser.parse_args()

    return args


def load_dataset (json_file):
    """
    Load the dataset, and create video_id to split and video to task mappings
    """

    with open(json_file) as json_file:
        COIN = json.load(json_file)['database']

    vid_split_map = {}
    vid_task_map = {}
    for vid_id, annotation in COIN.items():
        vid_split_map[vid_id] = annotation['subset']
        vid_task_map[vid_id] = int(annotation['recipe_type'])

    return COIN, vid_split_map, vid_task_map


def create_frames_folder (absolute_consistency_path, vid_task_map):
    """
    Create a folder to store all the frames of the full dataset
    """
    vid_id_to_task_map = dict()

    nested_frames_path = os.path.join(absolute_consistency_path, 'data/coin/raw_frames')
    frames_path = os.path.join(absolute_consistency_path, 'data/coin/all_frames')

    # Creating symbolic links for the respective frames
    if os.path.exists(frames_path):
        shutil.rmtree(frames_path) # Remove the existing soft links
    os.mkdir(frames_path)

    for task_folder in glob.glob(os.path.join(nested_frames_path,'*')):
        task_id = int(task_folder.split('/')[-1])
        for vid_frame_folder in glob.glob(os.path.join(task_folder, '*')):
            vid_id = vid_frame_folder.split('/')[-1]
            src_path = vid_frame_folder
            dst_path = os.path.join(frames_path, vid_id)
            os.symlink(src_path, dst_path)
            vid_id_to_task_map[vid_id] = task_id

    # Match both vid_task_maps
    for vid_id, task_id in vid_id_to_task_map.items():
        if vid_id not in vid_task_map:
            raise ValueError ("%s not in database, found with taskID %d" % (vid_id, task_id))
            continue
    
        if vid_task_map[vid_id] != task_id:
            raise ValueError ("for %s, taskID should be %d not %d" % (vid_id, task_id, 
                                vid_task_map[vid_id]))
            continue
 

def create_tag_file (in_tag_path, out_tag_path, frames_path, vid_task_map, prefix='', no_task=False, val_path=None):
    """
    Create the TAG proposal file correcting the shifted step_id error,
    adding task_id and splitting into validation set
    """

    # Read the input TAG file
    blocks = [b for b in read_block(in_tag_path)]

    # Do a frequency analysis of step ids in TAGs
    step_freq = dict()
    for block in blocks:
        for c in block['correct']:
            if c[0] in step_freq: step_freq[c[0]] += 1
            else: step_freq[c[0]] = 1

    train_step_freq = dict(); val_step_freq = dict()
    np.random.seed(0) # Deterministic random

    def determine_split(block):
        # Determine the train/val split of the given block
        if val_path is None:
            return ['train']

        split = None
        for c in block['correct']:
            step_id = c[0]
            if step_freq[step_id] <= 1:
                # Put this block in both sets
                split = ['train', 'val']
                break
            
            if step_id not in train_step_freq and step_id not in val_step_freq:
                # Offer to any set randomly
                if np.random.rand() > 0.5: split = ['train']
                else: split = ['val']
                break 

            if step_id not in train_step_freq:
                # Offer to train
                split = ['train']
                break

            if step_id not in val_step_freq:
                # Offer to val set
                split = ['val']
                break

        train_val_split = 0.2
        if split is None:
            # Randomly assign to one
            if np.random.rand() > train_val_split:
                split = ['train']
            else:
                split = ['val']

        # Update the dicts
        for c in block['correct']:
            step_id = c[0]
            if 'train' in split:
                if step_id in train_step_freq: train_step_freq[step_id] += 1
                else: train_step_freq[step_id] = 1
            if 'val' in split:
                if step_id in val_step_freq: val_step_freq[step_id] += 1
                else: val_step_freq[step_id] = 1

        return split

    # Write into out_tag file
    train_idx = val_idx = 1
    for block in blocks:
        split = determine_split(block)

        vid_id = block['id']
        task_id = vid_task_map[vid_id]
        modify_block_simple(block, frames_path, prefix, task_id)

        if len(split) == 2:
            print ("CAUTION: Writing VIDEO_ID: %s in both train and test set" % vid_id)
        
        if 'train' in split:
            write_block(block, out_tag_path, train_idx, no_task=no_task)
            train_idx += 1
        if 'val' in split:
            write_block(block, val_path, val_idx, no_task=no_task)
            val_idx += 1


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

    # Load the dataset
    json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN_full.json')
    COIN, vid_split_map, vid_task_map = load_dataset(json_path) 

    # Create the frames folder
    create_frames_folder(absolute_consistency_path, vid_task_map)

    # Creating the TAG files
    coin_path = os.path.join(absolute_consistency_path, 'data/coin')
    train_full_path = os.path.join(coin_path, 'coin_full_tag_train_proposal_list.txt')
    test_full_path = os.path.join(coin_path, 'coin_full_tag_test_proposal_list.txt')

    train_tag = os.path.join(coin_path, 'full/coin_tag_train_proposal_list.txt')
    test_tag = os.path.join(coin_path, 'full/coin_tag_test_proposal_list.txt')
    val_tag = os.path.join(coin_path, 'full/coin_tag_val_proposal_list.txt')

    frames_path = os.path.join(absolute_consistency_path, 'data/coin/all_frames')
    create_tag_file (test_full_path, test_tag,
                     frames_path=frames_path, vid_task_map=vid_task_map,
                     prefix=args.tag_prefix, no_task=args.no_task)
    create_tag_file (train_full_path, train_tag,
                     frames_path=frames_path, vid_task_map=vid_task_map,
                     prefix=args.tag_prefix, no_task=args.no_task,
                     val_path=val_tag)

    # Creating the belonging matrix
    step_to_task_map, W_matrix = create_W_matrix (COIN)
    step_to_task_map_path = os.path.join(coin_path, 'full/step_to_task_map.json')
    W_matrix_file = os.path.join(coin_path, 'full/W.npy')
    with open(step_to_task_map_path, 'w') as outfile:
        json.dump(step_to_task_map, outfile, indent=4)
    np.save(W_matrix_file, W_matrix)
    print ("Created W matrix")
