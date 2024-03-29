"""
Contains helper functions for interacting with the TAG files
"""
import os
import glob
import sys
import json

# Start reading
def read_block (tag_file):
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
            obj['correct'].append(f.readline().strip().split(' '))
        
        obj['preds'] = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            obj['preds'].append(f.readline().strip().split(' '))

        yield obj


def modify_block (block, subset_frames_path, prefix, step_mapping, task):
    id = block['id']
    block['path'] = os.path.join(prefix, id)
    new_frames = len(glob.glob(os.path.join(subset_frames_path, id, '*')))
    old_frames = block['frames']
    block['frames'] = new_frames
    scaling = new_frames / old_frames
    block['task'] = task

    for c in block['correct']:
        if c[0] not in step_mapping:
            step_mapping[c[0]] = str(len(step_mapping))
        c[0] = step_mapping[c[0]]

        c[1] = str(int(int(c[1]) * scaling))
        c[2] = str(int(int(c[2]) * scaling))

    for p in block['preds']:
        if p[0] not in step_mapping:
            step_mapping[p[0]] = str(len(step_mapping))
        p[0] = step_mapping[p[0]]

        p[3] = str(int(int(p[3]) * scaling))
        p[4] = str(int(int(p[4]) * scaling))


def modify_block_simple (block, frames_path, prefix, task_id):
    id = block['id']
    block['path'] = os.path.join(prefix, id)
    new_frames = len(glob.glob(os.path.join(frames_path, id, '*')))
    old_frames = block['frames']
    block['frames'] = new_frames
    scaling = new_frames / old_frames
    block['task'] = task_id

    for c in block['correct']:
        c0 = int(c[0])
        assert c0 == 0 or c0 > 1
        if c0 > 1:
            c[0] = str(c0 - 1)
        c[1] = str(int(int(c[1]) * scaling))
        c[2] = str(int(int(c[2]) * scaling))

    for p in block['preds']:
        p0 = int(p[0])
        assert p0 == 0 or p0 > 1
        if p0 > 1:
            p[0] = str(p0 - 1)
        p[3] = str(int(int(p[3]) * scaling))
        p[4] = str(int(int(p[4]) * scaling))


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
