"""
Trim the dataset
1. Remove unavailable videos
2. Remove classes appearing in 16 or less videos (lower 10th percentile of classes)
"""

import os, shutil
import glob
import json
import pandas as pd
import argparse

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Prepare the full dataset")
    parser.add_argument('--root', help='Path of the consistency repo', default='./')
    args = parser.parse_args()

    return args


def dataset_stats(COIN, frames_path, video_stats_path, class_stats_path, task_stats_path):
    """
    Print statistics of the dataset

    Video-wise
    1. Video ID
    2. Duration
    3. is_available
    4. Task_Name
    5. Task_ID
    6. Step List

    Step (class)-wise
    1. Class ID
    2. Class name
    3. Task Name
    4. Task ID
    5. #Planned videos
    6. #Available videos
    7. Blacklisted?
    8. New ID

    Task-wise
    1. Task ID
    2. Task Name
    3. Old #Classes
    4. New #Classes
    5. Old class list
    6. New class list
    """

    # Extract video stats
    video_stats = dict()
    for vid_id, vid_data in COIN.items():
        video_stats[vid_id] = dict()
        video_stats[vid_id]['duration'] = vid_data['duration']
        video_stats[vid_id]['is_available'] = os.path.exists(os.path.join(frames_path, vid_id))
        video_stats[vid_id]['task_name'] = vid_data['class']
        video_stats[vid_id]['task_id'] = vid_data['recipe_type']
        step_list = []
        for ann in vid_data['annotation']:
            step_list.append(ann['label'])
        video_stats[vid_id]['step_list'] = ', '.join(step_list)

    video_df = pd.DataFrame.from_dict(video_stats, orient='index')
    print (video_stats_path)
    video_df.to_csv(video_stats_path, sep=',', encoding='utf-8', index_label='Video ID')

    # Extract class stats
    VIDEO_THRESHOLD = 16
    class_stats = dict()
    for vid_id, vid_data in COIN.items():
        task_id = vid_data['recipe_type']
        task_name = vid_data['class']
        for ann in vid_data['annotation']:
            class_id = int(ann['id'])
            class_name = ann['label']
            # Current code will double count duplicate classes in a video
            # This is desired behavious as of now 
            if class_id in class_stats:
                assert class_stats[class_id]['class_name'] == class_name
                assert class_stats[class_id]['task_id'] == task_id
                assert class_stats[class_id]['task_name'] == task_name
            else:
                class_stats[class_id] = dict()
                class_stats[class_id]['class_name'] = class_name
                class_stats[class_id]['task_id'] = task_id
                class_stats[class_id]['task_name'] = task_name
                class_stats[class_id]['available_videos'] = 0
                class_stats[class_id]['planned_videos'] = 0

            class_stats[class_id]['planned_videos'] += 1
            if video_stats[vid_id]['is_available']:
                class_stats[class_id]['available_videos'] += 1

    new_id = 1
    for c in sorted(class_stats.keys()):
        if class_stats[c]['available_videos'] <= VIDEO_THRESHOLD:
            class_stats[c]['blacklist'] = True
            class_stats[c]['new_ID'] = 'NA'
        else:
            class_stats[c]['blacklist'] = False
            class_stats[c]['new_ID'] = str(new_id)
            new_id += 1

    class_df = pd.DataFrame.from_dict(class_stats, orient='index')
    class_df.sort_index(inplace=True)
    print (class_stats_path)
    class_df.to_csv(class_stats_path, sep=',', encoding='utf-8', index_label='Class ID')

    # Extract video stats
    task_stats = dict()
    for vid_id, vid_data in COIN.items():
        task_id = int(vid_data['recipe_type'])
        task_name = vid_data['class']

        if task_id in task_stats:
            assert task_stats[task_id]['task_name'] == task_name
        else:
            task_stats[task_id] = dict()
            task_stats[task_id]['task_name'] = task_name
            task_stats[task_id]['old_classes'] = set()
            task_stats[task_id]['new_classes'] = set()

        for ann in vid_data['annotation']:
            class_id = int(ann['id'])
            new_class_id = class_stats[class_id]['new_ID']
            task_stats[task_id]['old_classes'].add(class_id)
            if not class_stats[class_id]['blacklist']:
                task_stats[task_id]['new_classes'].add(new_class_id)

    for task_id in task_stats.keys():
        task_stats[task_id]['old_num_classes'] = len(task_stats[task_id]['old_classes'])
        task_stats[task_id]['new_num_classes'] = len(task_stats[task_id]['new_classes'])
        task_stats[task_id]['old_classes'] = ' '.join([str(c) for c in task_stats[task_id]['old_classes']])
        task_stats[task_id]['new_classes'] = ' '.join([str(c) for c in task_stats[task_id]['new_classes']])

    task_df = pd.DataFrame.from_dict(task_stats, orient='index')
    task_df.sort_index(inplace=True)
    print (task_stats_path)
    task_df.to_csv(task_stats_path, sep=',', encoding='utf-8', index_label='Task ID')


if __name__ == '__main__':
    args = parse_args()

    json_file = os.path.join(args.root, 'data/coin/COIN_full.json')
    with open(json_file) as json_file:
        COIN = json.load(json_file)['database']

    frames_path = os.path.join(args.root, 'data/coin/all_frames')
    video_stats_path = os.path.join(args.root, 'data/coin/full/vid_stats.csv')
    class_stats_path = os.path.join(args.root, 'data/coin/full/class_stats.csv')
    task_stats_path = os.path.join(args.root, 'data/coin/full/task_stats.csv')
    dataset_stats(COIN, frames_path, video_stats_path, class_stats_path, task_stats_path)
    