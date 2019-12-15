"""
Apply the COIN's TC approach over generated pickle files
"""

import pickle
import numpy as np
import argparse

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Enforce term consistency over the generated scores")
    parser.add_argument('pkl', help='Path of the result pkl file')
    parser.add_argument('--out_pkl', help='Path of the output pkl file')
    parser.add_argument('--W', help='Path to the W matrix', default='data/coin/W.npy')
    parser.add_argument('--pooling', help='How to pool. Should be either \'mean\' or \'max\'',
                        type=str, default='mean', choices=['mean', 'max'])
    
    return parser.parse_args()


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
    return e_x / e_x.sum(axis=dim, keepdims=True)


def enforce_TC(inpkl, outpkl, W_matrix, pooling='mean'):
    """
    Enforce task consistency on the generated scores using the W matrix
    """
    pool = np.mean if pooling == 'mean' else np.max
    W = np.load(W_matrix)

    old_results = pickle.load(open(inpkl, 'rb'))
    results = []
    for (proposals, act_scores, comp_scores, reg, *useless) in old_results:
        combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
        step_scores = pool(combined_scores, axis=0)
        step_scores = np.array([step_scores])

        one, K = step_scores.shape
        N, _ = proposals.shape
        assert (W.shape[0] == K and one == 1)

        # Predict video
        video_scores = np.matmul(step_scores, W)[0]
        video_prediction = np.argmax(video_scores)

        # Mask step scores
        mask = (W.T)[video_prediction]
        assert len(mask) == K

        mask = np.insert(mask, 0, 1)
        mask = 1 - mask
        mask = mask.astype(bool)
        assert len(mask) == K + 1 and mask.dtype == bool

        big_mask = np.array([mask] * N)
        assert (act_scores.shape == big_mask.shape == (N, K+1))
       
        act_scores[big_mask] = -100
        
        results.append((proposals, act_scores, comp_scores, reg, video_scores))
    
    pickle.dump(results, open(outpkl, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    inpkl = args.pkl
    outpkl = args.out_pkl
    W_matrix = args.W

    if outpkl is None:
        outpkl = inpkl.split('.pkl')[0] + '_tc.pkl'

    enforce_TC(inpkl, outpkl, W_matrix, pooling=args.pooling)

