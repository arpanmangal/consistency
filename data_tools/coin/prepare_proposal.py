"""
Map the full proposal file to the subset proposal file
"""
import os
import glob
import sys
import json

# Start reading
def read_block (file):
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
            obj['correct'].append(f.readline().strip().split(' '))
        obj['preds'] = []
        preds = int(f.readline().strip())
        for _p in range(preds):
            obj['preds'].append(f.readline().strip().split(' '))

        yield obj


def modify_block (block, subset_frames_path, mapping_dict):
    id = block['id']
    block['path'] = os.path.join(subset_frames_path, id)
    new_frames = len(glob.glob(os.path.join(subset_frames_path, id, '*')))
    old_frames = block['frames']
    block['frames'] = new_frames
    scaling = new_frames / old_frames

    for c in block['correct']:
        if c[0] not in mapping_dict:
            mapping_dict[c[0]] = str(len(mapping_dict))
        c[0] = mapping_dict[c[0]]

        c[1] = str(int(int(c[1]) * scaling))
        c[2] = str(int(int(c[2]) * scaling))

    for p in block['preds']:
        if p[0] not in mapping_dict:
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
    absolute_consistency_path = '/home/arpan/consistency' # Change this path for your machine
    subset_frames_path = os.path.join(absolute_consistency_path, 'data/coin/subset_frames')

    assert (len(sys.argv) >= 4)
    infile = sys.argv[1]; outfile = sys.argv[2]
    load_json = True if sys.argv[3] == 'y' else False
    prefix = '' if len(sys.argv) == 4 else subset_frames_path

    infile = os.path.join(absolute_consistency_path, 'data/coin/', infile)
    outfile = os.path.join(absolute_consistency_path, 'data/coin/', outfile)
    mapping_json_path = os.path.join(absolute_consistency_path, 'data/coin/proposal_mapping1.json')

    vid_ids = glob.glob(os.path.join(subset_frames_path, '*'))
    vid_ids = [id.split('/')[-1] for id in vid_ids]

    mapping_dict = {"0": "0"} if not load_json else json.load(open(mapping_json_path, 'r'))

    with open(outfile, 'w') as f:
        f.write('')

    blocks = read_block(infile)
    idx = 1
    for block in blocks:
        id = block['id']
        if id in vid_ids:
            modify_block(block, subset_frames_path, mapping_dict)
            write_block(block, outfile, idx)
            idx += 1

    print (load_json)
    if not load_json:
        with open(mapping_json_path, 'w', encoding='utf-8') as f:
            json.dump(mapping_dict, f, ensure_ascii=False, indent=2)
    else:
        print (json.dumps(mapping_dict, indent=2))

