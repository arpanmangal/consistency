"""
Converting the result file got by MMaction into the
format of SSN file
"""

import os
import sys
import pickle

def get_ids (file):
    """
    Parse blocks and read video ids
    """
    f = open(file, 'r')
    ids = []; ps = []
    while(len(f.readline()) > 0):
        id = f.readline().strip().split('/')[-1]
        ids.append(id)
        f.readline(); f.readline()
        corrects = int(f.readline().strip())
        for _c in range(corrects):
            f.readline()
        preds = int(f.readline().strip())
        ps.append(preds)
        for _p in range(preds):
            f.readline()

    return ids, ps


if __name__ == '__main__':
    absolute_consistency_path = '/home/cse/btech/cs1160321/scratch/BTP/consistency' # Change this path for your machine
    coin_path = os.path.join(absolute_consistency_path, 'data/coin/')
    test_tag = os.path.join(coin_path, 'coin_tag_test_proposal_list.txt')
    result_file = os.path.join(coin_path, 'result.pkl')

    assert (len(sys.argv) == 2)
    mm_result_file = sys.argv[1]

    vid_ids, ps = get_ids(test_tag)
    print (vid_ids)

    results = pickle.load(open(mm_result_file, 'rb'))

    assert len(vid_ids) == len(results) == len(ps)
    
    result = {}
    for id, p, r in zip(vid_ids, ps, results):
        r1, r2, r3, r4 = r
        n, k = r3.shape
        print (p, k)
        assert (n == p)
        result[id] = r

    print (len(results))
    print (len(results[0]))
    pickle.dump(result, open(result_file, 'wb'))


