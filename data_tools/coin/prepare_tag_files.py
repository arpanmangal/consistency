"""
Code to prepare full coin dataset
Trim the TAG files removing useless videos and low frequency classes

Combines the functionality of prepare_full_coin.py and trim_dataset.py
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

    vid_split_map = {}
    vid_task_map = {}
    for vid_id, annotation in COIN.items():
        vid_split_map[vid_id] = annotation['subset']
        vid_task_map[vid_id] = int(annotation['recipe_type'])

    return COIN, vid_split_map, vid_task_map


def create_frames_folder (old_frames, new_frames, vid_task_map):
    """
    Create a folder to store all the frames of the full dataset
    """
    vid_id_to_task_map = dict()

    nested_frames_path = old_frames
    frames_path = new_frames

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


def create_tag_file (in_tag_path, out_tag_path, frames_path, vid_task_map, prefix='', no_task=False, num_classes=0, vid_stats=None, class_stats=None):
    """
    Create the TAG proposal file correcting the shifted step_id error,
    adding task_id, and removing invalid videos
    """

    # Read the input TAG file
    # blocks = [b for b in read_block(in_tag_path)]

    # Write into out_tag file
    idx = 1
    with open(out_tag_path, 'w') as f:
        overwritten = True

    for block in read_block(in_tag_path, vid_stats):
        # print (idx)
        vid_id = block['id']
        task_id = vid_task_map[vid_id]
        modify_block_simple(block, frames_path, prefix, task_id, class_stats)
        write_block(block, out_tag_path, idx, no_task=no_task)
        idx += 1


# Start reading
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
            # assert vid_stats['is_available'][obj['id']]
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


def modify_block_simple (block, frames_path, prefix, task_id, class_stats):
    id = block['id']
    block['path'] = os.path.join(prefix, id)
    new_frames = len(glob.glob(os.path.join(frames_path, id, '*')))
    old_frames = block['frames']
    block['frames'] = new_frames
    scaling = new_frames / old_frames
    block['task'] = task_id

    remove_cidx = []
    for idx, c in enumerate(block['correct']):
        c0 = int(c[0])
        assert c0 == 0 or c0 > 1
        if c0 > 1:
            # Map to new class ID
            c0 = c0 - 1
            assert class_stats['task_id'][c0] == task_id
            if class_stats['blacklist'][c0]:
                assert np.isnan(class_stats['new_ID'][c0])
                remove_cidx.append(idx)
                c[0] = '0'
            else:
                c[0] = str(int(class_stats['new_ID'][c0]))

        c[1] = str(int(int(c[1]) * scaling))
        c[2] = str(int(int(c[2]) * scaling))

    # Remove deleted classes
    block['correct'] = [c for idx, c in enumerate(block['correct']) if idx not in remove_cidx]

    remove_pidx = []
    for p in block['preds']:
        p0 = int(p[0])
        assert p0 == 0 or p0 > 1
        if p0 > 1:
            # Map to new class ID
            p0 = p0 - 1
            assert class_stats['task_id'][p0] == task_id
            if class_stats['blacklist'][p0]:
                assert np.isnan(class_stats['new_ID'][p0])
                remove_pidx.append(idx)
                p[0] = '0'
            else:
                p[0] = str(int(class_stats['new_ID'][p0]))

        p[3] = str(int(int(p[3]) * scaling))
        p[4] = str(int(int(p[4]) * scaling))

    # Remove deleted classes
    block['preds'] = [p for idx, p in enumerate(block['preds']) if idx not in remove_pidx]

def write_block (block, ofile, idx, no_task=False):
    with open(ofile, 'a') as f:
        f.write('# %d\n' % idx)
        f.write('%s\n' % block['path'])
        f.write('%d\n' % block['frames'])
        if not no_task:
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

def create_W_matrix(task_stats, class_stats):
    """
    Create the belongs to matrix W
    (i, j)th entry is 1 if step i belongs to task j, otherwise 0
    """

    step_to_task_map = dict()
    task_to_step_map = dict()

    for oc in class_stats.index:
        if class_stats['blacklist'][oc]:
            assert np.isnan(class_stats['new_ID'][oc])
            continue
        new_class = int(class_stats['new_ID'][oc])
        task_id = int(class_stats['task_id'][oc])
        step_to_task_map[new_class] = task_id
        if task_id not in task_to_step_map:
            task_to_step_map[task_id] = set()
        task_to_step_map[task_id].add(new_class)

    # Formulate the same map again using task_stats and verify
    for task_id, steps in task_to_step_map.items():
        classes = set([int(e) for e in task_stats['new_classes'][task_id].split(' ')])
        assert len(steps.difference(classes)) == 0 and len(classes.difference(steps)) == 0
        task_to_step_map[task_id] = list(steps) # So as to store in .json

    W = np.zeros(
        (len(step_to_task_map.keys()), len(task_to_step_map.keys())),
        dtype=int
    )
    for step, task_id in step_to_task_map.items():
        assert 1 <= step <= W.shape[0]
        assert 0 <= task_id < W.shape[1]
        W[step-1][task_id] = 1

    return step_to_task_map, task_to_step_map, W


if __name__ == '__main__':
    root = '/home/cse/btech/cs1160321/consistency'
    newdatapath = os.path.join(root, 'data/mycoin')

    # Load the dataset
    json_path = os.path.join(root, 'data/coin/COIN_full.json')
    COIN, vid_split_map, vid_task_map = load_dataset(json_path)
    vid_stats = pd.read_csv(os.path.join(root, 'data/coin/new_full/vid_stats.csv'), index_col='Video ID') 
    task_stats = pd.read_csv(os.path.join(root, 'data/coin/new_full/task_stats.csv'), index_col='Task ID')
    class_stats = pd.read_csv(os.path.join(root, 'data/coin/new_full/class_stats.csv'), index_col='Class ID')

    # Create the frames folder
    oldframes = os.path.join(root, 'data/coin/raw_frames')
    newframes = os.path.join(newdatapath, 'rawframes')
    create_frames_folder(oldframes, newframes, vid_task_map)

    # Tag files
    coin_path = os.path.join(root, 'data/coin')
    train_full_path = os.path.join(coin_path, 'coin_full_tag_train_proposal_list.txt')
    test_full_path = os.path.join(coin_path, 'coin_full_tag_test_proposal_list.txt')

    trimmed_train = os.path.join(newdatapath, 'coin_tag_train_proposal_list.txt')
    trimmed_test = os.path.join(newdatapath, 'coin_tag_test_proposal_list.txt')

    num_classes = 780
    create_tag_file (train_full_path, trimmed_train,
                     frames_path=newframes, vid_task_map=vid_task_map,
                     num_classes=num_classes,
                     vid_stats=vid_stats, class_stats=class_stats)
    create_tag_file (test_full_path, trimmed_test,
                     frames_path=newframes, vid_task_map=vid_task_map,
                     num_classes=num_classes,
                     vid_stats=vid_stats, class_stats=class_stats)
    print ("Created TAG files")

    # Create W matrix
    step_to_task_map, task_to_step_map, W_matrix = create_W_matrix (task_stats, class_stats)
    step_to_task_map_path = os.path.join(newdatapath, 'step_to_task.json')
    task_to_step_map_path = os.path.join(newdatapath, 'task_to_step.json')
    W_matrix_file = os.path.join(newdatapath, 'W.npy')
    with open(step_to_task_map_path, 'w') as outfile:
        json.dump(step_to_task_map, outfile, indent=4)
    with open(task_to_step_map_path, 'w') as outfile:
        json.dump(task_to_step_map, outfile, indent=4)
    np.save(W_matrix_file, W_matrix)
    print ("Created W matrix")