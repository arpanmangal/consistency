"""
Script for predicting task using the step scores
"""

import argparse
import pickle
import json
import numpy as np
from task_head import Trainer, NMS

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train task model")
    subparsers = parser.add_subparsers(help='Mode of operation', dest='mode')

    # For pruning the proposals -- throw away too long / short proposals
    parser_prune = subparsers.add_parser('prune', help="Prune the proposals")
    parser_prune.add_argument('pkl', type=str, help="Path to the result pkl file")
    parser_prune.add_argument('save', type=str, help="Path to the generated pickle file")
    parser_prune.add_argument('--l', type=float, default=0.05, help="Low Range")
    parser_prune.add_argument('--h', type=float, default=0.6, help="Hi range")

    # For combining the result pickle file and TAG file
    parser_combine = subparsers.add_parser('combine', help="Combine the pkl and TAG files")
    parser_combine.add_argument('pkl', type=str, help="Path to result pkl file")
    parser_combine.add_argument('tag', type=str, help="Path to TAG file")
    parser_combine.add_argument('save', type=str, help="Path to generated pickle file")

    # For training a model
    parser_train = subparsers.add_parser('train', help="Train a model")
    parser_train.add_argument('config', type=str, help="Path to config file")
    parser_train.add_argument('pkl', type=str, help="Path to combined pickle file")
    parser_train.add_argument('save', type=str, help="Path to save the model")
    parser_train.add_argument('--lr', type=float, default=0.1, help="Learning Rate")
    parser_train.add_argument('--batch_size', type=int, default=64, help="Batch Size")
    parser_train.add_argument('--epochs', type=int, default=2, help="Number of Epochs")
    parser_train.add_argument('--decay', type=int, default=30, help="Number of Epochs after which to decay LR by 3")
    parser_train.add_argument('--log', type=str, required=True, help="Log File path")
    parser_train.add_argument('--pretrained', type=str, help="Path to pre-trained models")
    parser_train.add_argument('--validate', type=str, help="Path to validate pkl file for validation loss")
    parser_train.add_argument('--freq', type=int, default=5, help="Number of epochs after which to evaluate")
   
    # For testing model
    parser_test = subparsers.add_parser('test', help="Test a model")
    parser_test.add_argument('config', type=str, help="Path to config file")
    parser_test.add_argument('pkl', type=str, help="Path of combined pickle file")
    parser_test.add_argument('load', type=str, help="Path to saved model")

    # For doing NMS
    parser_nms = subparsers.add_parser('nms', help="Do NMS")
    parser_nms.add_argument('in_pkl', type=str, help="Pickle of combined scores")
    parser_nms.add_argument('out_pkl', type=str, help="Pickle of post-NMS combined scores")
    parser_nms.add_argument('--thres', type=float, default=0.2, help="NMS threshold")
    parser_nms.add_argument('--no_reg', default=False, action='store_true', help="Do not perform location regression on props")

    return parser.parse_args()


def prune(pkl, save, low, hi):
    """
    Prune the pickle scores, by removing too large / short proposals
    """
    print ('hola hola')
    pkl_data = pickle.load(open(pkl, 'rb'))

    results = []
    for data_idx, (props, act_scores, comp_scores, regs, useless) in enumerate(pkl_data):
        keep = []
        for idx, p in enumerate(props):
            if (low < p[1] - p[0] < hi):
                keep.append(idx)

        if len(keep) == 0:
            keep = [0] # Keep the first element
            print ('index {} is completely useless!!'.format(data_idx))
        results.append((props[keep, :], act_scores[keep, :], comp_scores[keep, :], regs[keep, :, :], useless))

    pickle.dump(results, open(save, 'wb'))


def combine(pkl_path, tag_path, save_path):
    """
    Combine the result scores with task IDs to generate dataset
    """
    def load_tag_file(filename):
        lines = list(open(filename))
        from itertools import groupby
        groups = groupby(lines, lambda x: x.startswith('#'))

        info_list = [[x.strip() for x in list(g)] for k, g in groups if not k]

        def parse_group(info):
            offset = 0
            vid = info[offset]
            vid_id = vid.split('/')[-1]
            offset += 1

            n_frame = int(float(info[1]) * float(info[3]))
            task_id = int(info[2])
            n_gt = int(info[4])
            offset = 5

            offset += n_gt
            n_pr = int(info[offset])

            return vid_id, task_id, n_frame, n_gt, n_pr

        return [parse_group(l) for l in info_list]

    scores = pickle.load(open(pkl_path, 'rb'))
    tags = load_tag_file(tag_path)

    assert len(scores) == len(tags)

    def _combine():
        def softmax(x, dim=1):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
            return e_x / e_x.sum(axis=dim, keepdims=True)

        for s, t in zip(scores, tags):
            props, act_scores, comp_scores, regs, _ = s
            _, task_id, _, _, n_pr = t

            assert props.shape[0] == n_pr

            combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
            
            yield props, combined_scores, regs, task_id

    results = [x for x in _combine()]
    pickle.dump(results, open(save_path, 'wb'))


def _parse_pkl(pkl_file):
    data = pickle.load(open(pkl_file, 'rb'))
    scores = []; task_ids = []; props = []
    for s in data:
        scores.append(s[1])
        task_ids.append(s[3])
        props.append(s[0])

    return scores, task_ids, props

def train(args):
    """
    Train a model
    """
    model_cfg = json.load(open(args.config, 'r'))
    train_cfg = {
        'lr': args.lr,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'decay': args.decay,
        'freq': args.freq,
        'log_file': args.log
    }

    print ("Generating model: {}".format(model_cfg['type']))
    print (model_cfg)
    trainer = Trainer(model_cfg)

    if args.pretrained is not None:
        print ("Loading pre-trained model from {}".format(args.pretrained))
        trainer.load_model(args.pretrained)
    print ('----------------------------------------------')

    scores, task_ids, props = _parse_pkl(args.pkl)
    if args.validate:
        val_scores, val_task_ids, val_props = _parse_pkl(args.validate)
        val_data = {
            'scores': val_scores,
            'task_ids': val_task_ids,
            'props': val_props
        }
    else:
        val_data = None

    trainer.train(train_cfg, scores, task_ids, props, val_data)
    print ("Save model to {}".format(args.save))
    trainer.save_model(args.save)


def test(pkl_path, load_path):
    """
    Test the trained model
    """ 
    model_cfg = json.load(open(args.config, 'r'))
    print ("Generating model: {}".format(model_cfg['type']))
    print (model_cfg)
    trainer = Trainer(model_cfg)

    print ("Loading trained model from {}".format(load_path))
    trainer.load_model(load_path)
    print ('----------------------------------------------')

    scores, task_ids, props = _parse_pkl(pkl_path)
    task_preds = trainer.predict(scores)

    task_ids = np.array(task_ids)
    print (task_ids)
    print (task_preds)
    print (np.sum(task_ids == task_preds))
    print (len(task_ids))
    print ("Accuracy: %.3f" % ((np.sum(task_ids == task_preds) / len(task_ids)) * 100))


def nms(in_pkl, out_pkl, thres, no_reg):
    """
    Do NMS over the combined scores
    """
    pkl_data = pickle.load(open(in_pkl, 'rb'))
    props = []; scores = []; regs = []; task_ids = []
    for data in pkl_data:
        p, s, r, t = data
        props.append(p)
        scores.append(s)
        regs.append(r)
        task_ids.append(t)

    nms = NMS(props, scores, regs)
    print ("Performing NMS...")
    print ("NMS Thres: {} | Reg: {}".format(thres, not no_reg))
    all_new_props, all_new_scores = nms.temporal_nms_gola(thres, no_reg)

    results = []
    for p, s, (_, _, _, t) in zip(all_new_props, all_new_scores, pkl_data):
        results.append((p, s, None, t))

    pickle.dump(results, open(out_pkl, 'wb'))


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'prune':
        assert 0 < args.l < args.h < 1
        prune(args.pkl, args.save, args.l, args.h)
    elif args.mode == 'combine':
        combine(args.pkl, args.tag, args.save)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args.pkl, args.load)
    elif args.mode == 'nms':
        nms(args.in_pkl, args.out_pkl, args.thres, args.no_reg)
    else:
        raise ValueError("Go Away")
