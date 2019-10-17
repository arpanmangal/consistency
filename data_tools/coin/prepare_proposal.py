"""
Map the full proposal file to the subset proposal file
"""
import os
import glob
import sys
import json
import argparse

# Start reading
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


def modify_block (block, subset_frames_path, prefix_path, mapping_dict, test_time=False):
    id = block['id']
    block['path'] = os.path.join(prefix_path, id)
    new_frames = len(glob.glob(os.path.join(subset_frames_path, id, '*')))
    old_frames = block['frames']
    block['frames'] = new_frames
    scaling = new_frames / old_frames

    for c in block['correct']:
        if not test_time and c[0] not in mapping_dict:
            mapping_dict[c[0]] = str(len(mapping_dict))
        c[0] = mapping_dict[c[0]]

        c[1] = str(int(int(c[1]) * scaling))
        c[2] = str(int(int(c[2]) * scaling))

    for p in block['preds']:
        if not test_time and p[0] not in mapping_dict:
            mapping_dict[p[0]] = str(len(mapping_dict))
        p[0] = mapping_dict[p[0]]

        p[3] = str(int(int(p[3]) * scaling))
        p[4] = str(int(int(p[4]) * scaling))


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate subset tag file from full tag file")
    parser.add_argument('--prefix', '-p', default=False, action='store_true', help="Whether path (to prepend to the video ID in frames path) should be empty")
    parser.add_argument('--ignore_background', '-i', default=False, action='store_true',
                        help="Whether to ignore the background class in test file")
    args = parser.parse_args()

    absolute_consistency_path = '/home/cse/btech/cs1160321/scratch/BTP/consistency' # Change this path for your machine
    subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')

    prefix = '' if args.prefix is True else subset_frames_path
    print ('prfix: ', prefix)

    coin_path = os.path.join(absolute_consistency_path, 'data/coin/')
    train_full_tag = os.path.join(coin_path, 'coin_full_tag_train_proposal_list.txt')
    test_full_tag = os.path.join(coin_path, 'coin_full_tag_test_proposal_list.txt')
    train_tag = os.path.join(coin_path, 'coin_tag_train_proposal_list.txt')
    test_tag = os.path.join(coin_path, 'coin_tag_test_proposal_list.txt')
    mapping_json_path = os.path.join(coin_path, 'proposal_mapping.json')

    vid_ids = glob.glob(os.path.join(subset_frames_path, '*'))
    vid_ids = [id.split('/')[-1] for id in vid_ids]


    # Train file
    infile = train_full_tag; outfile = train_tag

    mapping_dict = {"0": "0"} 

    with open(outfile, 'w') as f:
        f.write('')

    blocks = read_block(infile)
    idx = 1
    for block in blocks:
        id = block['id']
        if id in vid_ids:
            modify_block(block, subset_frames_path, prefix, mapping_dict)
            write_block(block, outfile, idx)
            idx += 1
     
    # Test file
    infile = test_full_tag; outfile = test_tag

    with open(outfile, 'w') as f:
        f.write('')

    blocks = read_block(infile, ignore_background=args.ignore_background)
    idx = 1
    for block in blocks:
        id = block['id']
        if id in vid_ids:
            modify_block(block, subset_frames_path, prefix, mapping_dict, test_time=True)
            write_block(block, outfile, idx)
            idx += 1

    with open(mapping_json_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

