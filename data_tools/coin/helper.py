"""
Contains helper functions for interacting with the TAG files
"""
import os
import glob
import sys
import json

# Start reading
def read_block (tag_file, tag_pruning_thres=0.6):
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
        
        props = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            props.append(f.readline().strip().split(' '))

        # Have a minimum of two proposals
        obj['preds'] = props
        # for prop in props:
        #     if (float(prop[-1]) - float(prop[-2])) / obj['frames'] < tag_pruning_thres:
        #         obj['preds'].append(prop)

        # if len(obj['preds']) < 2:
        #     obj['preds'] = props

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
