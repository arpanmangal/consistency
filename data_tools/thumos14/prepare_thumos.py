"""
Modify the THUMOS TAG files 
1. Remove useless videos
2. Add task label
"""

import os
import glob

def load_blocks(tag_file):
    """
    Load the TAG blocks which are valid
    """
    blocks = []
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

        if corrects > 0:
            # A valid video
            blocks.append(obj)

    return blocks


def modify_block (block, frames_path, task_id=None):
    id = block['id']
    new_frames = len(glob.glob(os.path.join(frames_path, id, 'img_*')))
    old_frames = block['frames']
    assert old_frames == 1

    block['frames'] = new_frames
    if task_id is not None:
        block['task'] = task_id

    for c in block['correct']:
        c[1] = str(int(float(c[1]) * new_frames))
        c[2] = str(int(float(c[2]) * new_frames))

    for p in block['preds']:
        p[3] = str(int(float(p[3]) * new_frames))
        p[4] = str(int(float(p[4]) * new_frames))

    return block


def write_blocks(blocks, out_tag_file, no_task=True):
    """
    Write the blocks to the file
    """
    f = open(out_tag_file, 'w')
    for idx, block in enumerate(blocks, 1):
        f.write('# %d\n' % idx)
        f.write('%s\n' % block['id'])
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

    f.close()


data_root = 'data/thumos14'
rawframes = os.path.join(data_root, 'rawframes')
train_tag = os.path.join(data_root, 'thumos14_tag_val_normalized_proposal_list.txt')
out_train_tag = os.path.join(data_root, 'thumos14_tag_val_proposal_list.txt')
test_tag = os.path.join(data_root, 'thumos14_tag_test_normalized_proposal_list.txt')
out_test_tag = os.path.join(data_root, 'thumos14_tag_test_proposal_list.txt')

write_blocks(
    [modify_block(block, rawframes, task_id=0) for block in load_blocks(train_tag)],
    out_train_tag,
    no_task=False
)

write_blocks(
    [modify_block(block, rawframes, task_id=0) for block in load_blocks(test_tag)],
    out_test_tag,
    no_task=False
)