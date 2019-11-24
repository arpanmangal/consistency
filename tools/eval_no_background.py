"""
Script to evaluate the pkl file while not considering the background class
"""

import os
import shutil
import shlex
import argparse
import json
import subprocess
import pickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Test with or without the background class')
    parser.add_argument('tag_file', help='path of the TAG file')
    parser.add_argument('pkl_file', help='path of the pkl file')
    parser.add_argument('config_file', help='path of the config file')
    parser.add_argument('--use_background', action='store_true', help='Use the 0 background class')

    return parser.parse_args()

def read_block (file):
    f = open(file, 'r')
    while (len(f.readline()) > 0):
        # Keep reading the block
        background_indices = []
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

        obj['preds'] = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            p_tuple = f.readline().strip().split(' ')
            if p_tuple[0] == '0':
                background_indices.append(_p)
                continue
            obj['preds'].append(p_tuple)

        yield len(obj['preds']), background_indices

def evaluate(tag_file, pkl_file, config_file, use_background=False):
    """
    Read the TAG and pkl file, make new pkl file for without zero
    Run the eval localizer
    """

    if not use_background:
        results = pickle.load(open(pkl_file, 'rb'))
        new_results = []
        for (p, background_indices), r in zip(read_block(tag_file), results):
            # Write the results, removing the background class
            r1, r2, r3, r4 = r
            r1 = np.delete(r1, background_indices, 0)
            r2 = np.delete(r2, background_indices, 0)
            r3 = np.delete(r3, background_indices, 0)
            r4 = np.delete(r4, background_indices, 0)
            r = (r1, r2, r3, r4)

            # Both should have exactly the same number of proposals
            n, k = r3.shape
            assert (n == p)

            new_results.append(r)

        new_pkl = os.path.join(os.path.dirname(pkl_file), '_result_without0.pkl')
        with open(new_pkl, 'wb') as f:
            pickle.dump(new_results, f)
    
    else:
        new_pkl = pkl_file

    # Final Evaluation
    command = "python3 tools/eval_localize_results.py %s %s --eval coin" % (config_file, new_pkl)
    command = shlex.split(command)
    print(command)
    process = subprocess.Popen(command)
    process.wait()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.tag_file, args.pkl_file, args.config_file, use_background=args.use_background)
