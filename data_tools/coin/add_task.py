"""
Script for adding task ID to the TAG files
Does not try to change class IDs
"""

import os, glob
import shutil
import json
import pandas as pd
import numpy as np

def load_dataset (json_file):
    """
    Load the dataset, and create video_id to split and video to task mappings
    """

    with open(json_file) as json_file:
        COIN = json.load(json_file)['database']

    vid_task_map = dict()
    step_task_map = dict()
    for vid_id, data in COIN.items():
        task_id = int(data['recipe_type'])
        vid_task_map[vid_id] = task_id
        for ann in data['annotation']:
            step_id = int(ann['id'])
            if step_id in step_task_map:
                assert step_task_map[step_id] == task_id
            else:
                step_task_map[step_id] = task_id

    return COIN, vid_task_map, step_task_map


def create_tag_file (in_tag_path, out_tag_path, frames_path, vid_stats, vid_task_map, step_task_map, prefix=''):
    """
    Create the TAG proposal file correcting the shifted step_id error,
    adding task_id, and removing invalid videos
    """
    def read_block (tag_file, vid_stats):
        print ('Processing %s TAG file' % tag_file)
        f = open(tag_file, 'r')
        while (len(f.readline()) > 0):
            # Keep reading the block
            obj = {}
            obj['id'] = f.readline().strip().split('/')[-1]
            obj['frames'] = int(f.readline().strip())
            obj['useless'] = f.readline()
            obj['correct'] = []
            corrects = int(f.readline().strip())
            for _c in range(corrects):
                annotation = f.readline().strip().split(' ')
                obj['correct'].append(annotation)
            obj['preds'] = []
            preds = int(f.readline().strip())
            for _p in range(preds):
                obj['preds'].append(f.readline().strip().split(' '))
            if len(obj['correct']) == 0 or not vid_stats['is_available'][obj['id']]:
                # Invalid video
                # print ('Video # %s is INVALID' % obj['id'])
                continue
            yield obj

    def modify_block_simple (block, task_id):
        id = block['id']
        block['path'] = os.path.join(prefix, id)
        new_frames = len(glob.glob(os.path.join(frames_path, id, '*')))
        old_frames = block['frames']
        block['frames'] = new_frames
        scaling = new_frames / old_frames
        block['task'] = task_id

        for idx, c in enumerate(block['correct']):
            c0 = int(c[0])
            assert c0 == 0 or c0 > 1
            if c0 > 1:
                # Check task ID
                assert step_task_map[c0 - 1] == task_id
            c[1] = str(int(int(c[1]) * scaling))
            c[2] = str(int(int(c[2]) * scaling))

        for idx, p in enumerate(block['preds']):
            p0 = int(p[0])
            assert p0 == 0 or p0 > 1
            if p0 > 1:
                # Map to new class ID
                # Check task ID
                assert step_task_map[p0 - 1] == task_id
            p[3] = str(int(int(p[3]) * scaling))
            p[4] = str(int(int(p[4]) * scaling))

    def write_block (block, ofile, idx):
        with open(ofile, 'a') as f:
            f.write('# %d\n' % idx)
            f.write('%s\n' % block['path'])
            f.write('%d\n' % block['frames'])
            f.write('%d\n' % block['task'])
            f.write(block['useless'])
            corrects = block['correct']
            preds = block['preds']
            f.write('%d\n' % len(corrects))
            for c in corrects:
                f.write('%s\n' % ' '.join(c))
            f.write('%d\n' % len(preds))
            for p in preds:
                f.write('%s\n' % ' '.join(p))

    with open(out_tag_path, 'w') as f:
        overwritten = True

    idx = 1
    for block in read_block(in_tag_path, vid_stats):
        # Read the input TAG file
        # print (idx)
        vid_id = block['id']
        task_id = vid_task_map[vid_id]
        modify_block_simple(block, task_id)
        if len(block['correct']) == 0: raise ValueError
        # Write into out_tag file
        write_block(block, out_tag_path, idx)
        idx += 1


if __name__ == '__main__':
    root = '/home/cse/btech/cs1160321/consistency'
    newdatapath = os.path.join(root, 'data/tcoin')

    # Load the dataset
    json_path = os.path.join(root, 'data/coin/COIN_full.json')
    vid_stats = pd.read_csv(os.path.join(root, 'data/coin/new_full/vid_stats.csv'), index_col='Video ID') 
    newframes = os.path.join(newdatapath, 'rawframes')
    COIN, vid_task_map, step_task_map = load_dataset(json_path)

    # Tag files
    coin_path = os.path.join(root, 'data/coin')
    train_full_path = os.path.join(coin_path, 'coin_full_tag_train_proposal_list.txt')
    test_full_path = os.path.join(coin_path, 'coin_full_tag_test_proposal_list.txt')

    new_train = os.path.join(newdatapath, 'coin_tag_train_proposal_list.txt')
    new_test = os.path.join(newdatapath, 'coin_tag_test_proposal_list.txt')

    create_tag_file (train_full_path, new_train,
                    frames_path=newframes, vid_stats=vid_stats,
                    vid_task_map=vid_task_map, step_task_map=step_task_map)
    create_tag_file (test_full_path, new_test,
                    frames_path=newframes, vid_stats=vid_stats,
                    vid_task_map=vid_task_map, step_task_map=step_task_map)

    
