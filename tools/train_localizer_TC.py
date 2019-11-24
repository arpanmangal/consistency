from __future__ import division

import os
import shutil
import shlex
import argparse
import json
import subprocess
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train an action localizer with TC')
    parser.add_argument('data_path', help='path to data containing the JSON and TAG files')
    parser.add_argument('base_config', help='base config file path')
    parser.add_argument('work_dir', help='the dir to save logs and models')
    parser.add_argument('mode', type=str, choices=['setup', 'train', 'test', 'all', 'eval'])
    parser.add_argument('--model', default='latest.pth', help='name of the pth file to use for testing')
    parser.add_argument('--out_pkl', default='result.pkl', help='name of the pth file to use for testing')
    parser.add_argument('--no_background', action='store_true', help='Remove the background class (0) proposals')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()
    
    return args


def create_task_list_n_dirs(args):
    """
    Returns a tuple of (COIN dataset, set of task_ids, task_dirs)
    """
    # Path of various files
    json_file = os.path.join(args.data_path, 'COIN.json')

    # First parse the JSON path
    with open(json_file) as json_file:
        COIN = json.load(json_file)['database']
    task_ids = set()
    for v_id, vid in COIN.items():
        task_ids.add(vid['recipe_type'])

    # Different folders for different task training
    dirs = dict()
    for task in task_ids:
        task_dir = os.path.join(args.work_dir, str(task))
        dirs[task] = task_dir

    return COIN, task_ids, dirs


def setup(args):
    # Generating the task IDs and folders
    COIN, task_ids, dirs = create_task_list_n_dirs(args)

    # Create different folders for different task training
    for task, task_dir in dirs.items():
        try:
            shutil.rmtree(task_dir)
        except:
            pass
        os.makedirs(task_dir)

    # Creating mapping files
    task_step_ids = dict() # Containing set of step_ids corresponding to each task
    for task in task_ids:
        task_step_ids[task] = set()
    for vid_id, vid in COIN.items():
        task_id = vid['recipe_type']
        for ann in vid['annotation']:
            task_step_ids[task_id].add(ann['id'])

    overall_indv_mapping = dict()
    indv_overall_mapping = dict()
    for task in task_ids:
        indv_overall_mapping[task] = dict()

    for task, ts_ids in task_step_ids.items():
        for idx, step_id in enumerate(ts_ids):
            new_id = str(idx + 1)
            indv_overall_mapping[task][new_id] = step_id
            overall_indv_mapping[step_id] = new_id

    for task, task_dir in dirs.items():
        mapping_file = os.path.join(task_dir, 'mapping.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(indv_overall_mapping[task], f, ensure_ascii=False, indent=2, sort_keys=True)

    overall_mapping_file = os.path.join(args.work_dir, 'overall_mapping.json')
    with open(overall_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(overall_indv_mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    
    # Setup the config files
    for task, task_dir in dirs.items():
        config_file = os.path.join(task_dir, 'config.py')

        with open(args.base_config, 'r') as f:
            lines = f.readlines()

        with open(config_file, 'w') as f:
            for line in lines:
                if line[:13] == 'ann_file_root':
                    line = "ann_file_root = '%s/'\n" % task_dir 
                elif line[:19] == '        num_classes':
                    line = '        num_classes=%d,\n' % len(task_step_ids[task])
                f.write(line)

    # Setting up the tag file
    tag_vid_count = dict()
    for task, task_dir in dirs.items():
        tag_vid_count[task] = 0
        tag_train_file = os.path.join(task_dir, 'coin_tag_train_proposal_list.txt')
        tag_test_file = os.path.join(task_dir, 'coin_tag_test_proposal_list.txt')
        with open(tag_train_file, 'w') as f:
            status = 'File created'
        with open(tag_test_file, 'w') as f:
            status = 'File created'
    
    # Process the tag files
    def process_tag_file(tag_file_name):
        tag_file = os.path.join(args.data_path, tag_file_name)
        for block in read_block(tag_file):
            id = block['id']
            task_id = COIN[id]['recipe_type']
            task_tag_file = os.path.join(dirs[task_id], tag_file_name)
            tag_vid_count[task_id] += 1

            modify_block(block, overall_indv_mapping, task_step_ids[task_id])
            write_block(block, task_tag_file, tag_vid_count[task_id])

    process_tag_file('coin_tag_train_proposal_list.txt')
    process_tag_file('coin_tag_test_proposal_list.txt')


def read_block (file, ignore_background=False):
    f = open(file, 'r')
    while (len(f.readline()) > 0):
        # Keep reading the block
        obj = {}
        obj['id'] = f.readline().strip().split('/')[-1]
        obj['frames'] = int(f.readline().strip())
        obj['useless'] = f.readline()
        obj['correct'] = []
        corrects = int(f.readline().strip())
        for _c in range(corrects):
            c_tuple = f.readline().strip().split(' ')
            if c_tuple[0] == '0' and ignore_backgound:
                continue
            obj['correct'].append(c_tuple)
            #obj['correct'].append(f.readline().strip().split(' '))
        obj['preds'] = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            p_tuple = f.readline().strip().split(' ')
            if p_tuple[0] == '0' and ignore_background:
                continue
            obj['preds'].append(p_tuple)
            #obj['preds'].append(f.readline().strip().split(' '))

        yield obj


def modify_block (block, mapping_dict, task_step_ids):
    block['path'] = block['id']

    for c in block['correct']:
        step_id = c[0]
        if step_id not in task_step_ids:
            raise ValueError("Gold contains inconsistent data")
        else:
            c[0] = mapping_dict[step_id]

    for p in block['preds']:
        step_id = p[0]
        if step_id not in task_step_ids:
            p[0] = "0"
        else:
            p[0] = mapping_dict[step_id]


def modify_block_rev (block, indv_overall_mapping, remove_background=False):
    background_indices = []
    block['path'] = block['id']

    for c in block['correct']:
        if c[0] != '0':
            c[0] = indv_overall_mapping[c[0]]

    for idx, p in enumerate(block['preds']):
        if p[0] != '0':
            p[0] = indv_overall_mapping[p[0]]
        else:
            background_indices.append(idx)

    if remove_background:
        # Throw away the proposals with 0 class
        block['preds'] = np.delete(block['preds'], background_indices, 0)

    return len(block['preds']), background_indices
    



def write_block (block, ofile, idx):
    with open(ofile, 'a') as f:
        f.write('# %d\n' % idx)
        f.write('%s\n' % block['path'])
        f.write('%d\n' % block['frames'])
        f.write(block['useless'])

        corrects = block['correct']
        preds = block['preds']

        f.write('%d\n' % len(corrects))
        for c in corrects:
            f.write('%s\n' % ' '.join(c))

        f.write('%d\n' % len(preds))
        for p in preds:
            f.write('%s\n' % ' '.join(p))


# Training code
def train(args):
    # Generating the task IDs and folders
    COIN, task_ids, dirs = create_task_list_n_dirs(args)

    for task, task_dir in dirs.items():
        config_file = os.path.join(task_dir, 'config.py')
        command = "/usr/bin/bash tools/dist_train_localizer.sh %s %d --work_dir %s" % (config_file, args.gpus, task_dir)
        command = shlex.split(command)
        print(command)
        process = subprocess.Popen(command)
        process.wait()


def test(args):
    # Generating the task IDs and folders
    COIN, task_ids, dirs = create_task_list_n_dirs(args)

    for task, task_dir in dirs.items():
        config_file = os.path.join(task_dir, 'config.py')
        pth_file = os.path.join(task_dir, args.model)
        out_pkl = os.path.join(task_dir, args.out_pkl)
        command = "python3 tools/test_localizer.py %s %s --gpus %d --out %s --eval coin" % (config_file, pth_file, args.gpus, out_pkl)
        command = shlex.split(command)
        print(command)
        process = subprocess.Popen(command)
        process.wait()


def evaluate(args):
    """
    Combine the scores of all the pkl files
    Combine all the tag files
    Evaluate
    """
    # Generating the task IDs and folders
    COIN, task_ids, dirs = create_task_list_n_dirs(args)

    tag_test_file = os.path.join(args.work_dir, 'coin_tag_test_proposal_list.txt')
    with open(tag_test_file, 'w') as f:
        status = 'File created'

    overall_mapping_file = os.path.join(args.work_dir, 'overall_mapping.json')
    with open(overall_mapping_file, 'r') as f:
        overall_mapping = json.load(f)

    K = len(overall_mapping) # Num of step classes
    remove_background = args.no_background

    all_results = []

    vid_count = 0
    for task, task_dir in dirs.items():
        task_tag_test = os.path.join(task_dir, 'coin_tag_test_proposal_list.txt')
        out_pkl = os.path.join(task_dir, args.out_pkl)
        mapping_file = os.path.join(task_dir, 'mapping.json')
        with open(mapping_file) as jf:
            indv_overall_mapping = json.load(jf)

        results = pickle.load(open(out_pkl, 'rb'))
        for block, r in zip(read_block(task_tag_test), results):
            # Write the block
            vid_count += 1
            p, background_indices = modify_block_rev (block, indv_overall_mapping, remove_background=remove_background)
            write_block(block, tag_test_file, vid_count)

            # Write the results
            r1, r2, r3, r4 = r
            if remove_background:
                # Remove the background classes
                r1 = np.delete(r1, background_indices, 0)
                r2 = np.delete(r2, background_indices, 0)
                r3 = np.delete(r3, background_indices, 0)
                r4 = np.delete(r4, background_indices, 0)
                r = (r1, r2, r3, r4)

            # Both should have exactly the same number of proposals
            n, k = r3.shape
            assert (n == p)

            N = n
            m1 = r1 # Proposals are same as before
            m2 = np.ones((N, K + 1)) * -100 # Giving a large negative score to the non useful ones
            m3 = np.ones((N, K)) * -100 # Giving a large negative score to the non useful ones
            m4 = np.zeros((N, K, 2)) # All regression parameters zero

            m2[:, 0] = r2[:, 0]
            for ind_id, overall_id in indv_overall_mapping.items():
                ind_id = int(ind_id)
                overall_id = int(overall_id)
                
                m2[:, overall_id] = r2[:, ind_id]
                m3[:, overall_id - 1] = r3[:, ind_id - 1]
                m4[:, overall_id - 1, :] = r4[:, ind_id - 1, :]

            m = m1, m2, m3, m4
            all_results.append(m)

    final_result = os.path.join(args.work_dir, 'final_result_with0.pkl')
    with open(final_result, 'wb') as f:
        pickle.dump(all_results, f)

    # Final Evaluation
    config_file = os.path.join(args.work_dir, 'config.py')
    command = "python3 tools/eval_localize_results.py %s %s --eval coin" % (config_file, final_result)
    command = shlex.split(command)
    print(command)
    process = subprocess.Popen(command)
    process.wait()



if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'setup':
        setup(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'eval':
        evaluate(args)
    # elif args.mode == 'all':
    #     setup(args)
    #     train(args)