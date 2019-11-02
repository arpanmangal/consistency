from __future__ import division

import os
import shutil
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Train an action localizer with TC')
    parser.add_argument('data_path', help='path to data containing the JSON and TAG files')
    parser.add_argument('base_config', help='base config file path')
    parser.add_argument('work_dir', help='the dir to save logs and models')
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


def main():
    args = parse_args()

    # Path of various files
    json_file = os.path.join(args.data_path, 'COIN.json')
    tag_train_file = os.path.join(args.data_path, 'coin_tag_train_proposal_list.txt')
    tag_test_file = os.path.join(args.data_path, 'coin_tag_test_proposal_list.txt')

    # First parse the JSON path
    with open(json_file) as json_file:
        COIN = json.load(json_file)['database']
    task_ids = set()
    for v_id, vid in COIN.items():
        task_ids.add(vid['recipe_type'])

    # Create different folders for different task training
    dirs = dict()
    for task in task_ids:
        task_dir = os.path.join(args.work_dir, str(task))
        shutil.rmtree(task_dir)
        os.makedirs(task_dir)
        dirs[task] = task_dir

    # Setup the config files
    for task, task_dir in dirs.items():
        config_file = os.path.join(task_dir, 'config.py')

        with open(args.base_config, 'r') as f:
            lines = f.readlines()

        with open(config_file, 'w') as f:
            for line in lines:
                if line[:13] == 'ann_file_root':
                    line = 'ann_file_root = %s\n' % task_dir 
                f.write(line)

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
            idx = str(idx)
            indv_overall_mapping[task][idx] = step_id
            overall_indv_mapping[step_id] = idx

    for task, task_dir in dirs.items():
        mapping_file = os.path.join(task_dir, 'mapping.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(indv_overall_mapping[task], f, ensure_ascii=False, indent=2, sort_keys=True)

    overall_mapping_file = os.path.join(args.work_dir, 'overall_mapping.json')
    with open(overall_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(overall_indv_mapping, f, ensure_ascii=False, indent=2, sort_keys=True)
    

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
    # def process_tag_file(tag_file_name):


if __name__ == '__main__':
    main()