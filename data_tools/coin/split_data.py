import os
import json
import numpy as np
import argparse

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Split the training set into training, validation and test set")
    parser.add_argument('--root', help='Path of the consistency repo', default='/home/arpan/BTP/consistency')
    parser.add_argument('--no_val', help='No validation split', action='store_true', default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    absolute_consistency_path = args.root

    json_path = os.path.join(absolute_consistency_path, 'data/coin/COIN_full.json')
    json_split_path = os.path.join(absolute_consistency_path, 'data/coin/COIN_full_split.json')

    # Split the current dataset into 15% validation, 20% test and 65% training set
    val_size = 0.15; test_size = 0.20; train_size = 1 - val_size - test_size
    np.random.seed(0) # Deterministically random

    # Reading the dataset
    with open(json_path) as json_file:
        COIN = json.load(json_file)['database']

    for vid_id, annotation in COIN.items():
        random_no = np.random.rand()
        if random_no < test_size:
            annotation['subset'] = 'testing'
        elif (not args.no_val) and random_no < test_size + val_size:
            annotation['subset'] = 'validation'
        else:
            annotation['subset'] = 'training'

    # Write the dataset
    coin_split = {'database': COIN}
    with open(json_split_path, 'w') as outfile:
        json.dump(coin_split, outfile, indent=4)

