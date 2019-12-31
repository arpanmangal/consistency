"""
Script for predicting task using the step scores
"""

import sys, os, glob, re
import time
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

    # For combining the result pickle file and TAG file
    parser_combine = subparsers.add_parser('combine', help="Combine the pkl and TAG files")
    parser_combine.add_argument('pkl', type=str, help="Path to result pkl file")
    parser_combine.add_argument('tag', type=str, help="Path to TAG file")
    parser_combine.add_argument('save', type=str, help="Path to generated pickle file")

    # For doing NMS
    parser_nms = subparsers.add_parser('nms', help="Do NMS")
    parser_nms.add_argument('in_pkl', type=str, help="Pickle of combined scores")
    parser_nms.add_argument('out_pkl', type=str, help="Pickle of post-NMS combined scores")
    parser_nms.add_argument('--thres', type=float, default=0.2, help="NMS threshold")
    parser_nms.add_argument('--no_reg', default=False, action='store_true', help="Do not perform location regression on props")

    # For training a model
    parser_train = subparsers.add_parser('train', help="Train a model")
    parser_train.add_argument('work_dir', type=str, help="Path of work directory. Should contain a config.py")
    parser_train.add_argument('pkl', type=str, help="Path to combined pickle file to be used for training")
    parser_train.add_argument('--validate', type=str, help="Path to validate pkl file for validation loss")
   
    # For testing model
    parser_test = subparsers.add_parser('test', help="Test a model")
    parser_test.add_argument('work_dir', type=str, help="Path of work directory. Should contain a config.py")
    parser_test.add_argument('pkl', type=str, help="Path of combined pickle file")
    parser_test.add_argument('load', type=int, help="Epoch number of the saved model")

    # For testing multiple model checkpoints
    parser_multitest = subparsers.add_parser('multitest', help="Test multiple model checkpoints")
    parser_multitest.add_argument('work_dir', type=str, help="Path of work directory. Should contain a config.py")
    parser_multitest.add_argument('pkl', type=str, help="Path of combined pickle file")
    parser_multitest.add_argument('log_file', type=str, help="Name of log file corresponding to the model")

    return parser.parse_args()


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

            # assert props.shape[0] == n_pr

            combined_scores = softmax(act_scores[:, 1:]) * np.exp(comp_scores)
            
            yield props, combined_scores, regs, task_id

    results = [x for x in _combine()]
    pickle.dump(results, open(save_path, 'wb'))

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
    all_new_props, all_new_scores = nms.temporal_nms(thres, no_reg)

    results = []
    for p, s, (_, _, _, t) in zip(all_new_props, all_new_scores, pkl_data):
        results.append((p, s, None, t))

    pickle.dump(results, open(out_pkl, 'wb'))


def _parse_pkl(pkl_file):
    data = pickle.load(open(pkl_file, 'rb'))
    scores = []; task_ids = []; props = []
    for s in data:
        scores.append(s[1])
        task_ids.append(s[3])
        props.append(s[0])

    return scores, task_ids, props

def train(work_dir, pkl, validate=None):
    """
    Train a model
    """
    sys.path.append(work_dir)
    from config import model_cfg, train_cfg

    for attr in ['type', 'num_steps', 'num_tasks']:
        assert attr in model_cfg

    for attr in ['lr', 'batch_size', 'epochs', 'decay', 'freq']:
        assert attr in train_cfg

    print ("Generating model: {}".format(model_cfg['type']))
    trainer = Trainer(model_cfg)

    if 'pretrained' in train_cfg:
        print ("Loading pre-trained model from {}".format(train_cfg['pretrained']))
        trainer.load_model(train_cfg['pretrained'])
    print ('----------------------------------------------')

    scores, task_ids, props = _parse_pkl(pkl)
    train_data = {
        'scores': scores,
        'task_ids': task_ids,
        'props': props
    }
    if validate:
        val_scores, val_task_ids, val_props = _parse_pkl(validate)
        val_data = {
            'scores': val_scores,
            'task_ids': val_task_ids,
            'props': val_props
        }
    else:
        val_data = None

    trainer.train(train_cfg, work_dir, train_data, val_data=val_data)

def test(work_dir, pkl_path, load_epoch, silent=False):
    """
    Test the trained model
    """
    load_path = os.path.join(work_dir, 'epoch_{}.pth'.format(load_epoch))
    sys.path.append(work_dir)
    from config import model_cfg

    for attr in ['type', 'num_steps', 'num_tasks']:
        assert attr in model_cfg
    if not silent: print ("Generating model: {}".format(model_cfg['type']))
    trainer = Trainer(model_cfg)

    if not silent: 
        print (model_cfg)
        print ("Loading trained model from {}".format(load_path))
        print ('----------------------------------------------')
    trainer.load_model(load_path)

    scores, task_ids, props = _parse_pkl(pkl_path)
    task_preds = trainer.predict(scores, props)

    task_ids = np.array(task_ids)
    accuracy = (np.sum(task_ids == task_preds) / len(task_ids)) * 100
    if not silent: 
        print (task_ids)
        print (task_preds)
        print (np.sum(task_ids == task_preds))
        print (len(task_ids))
        print ("Accuracy: %.3f" % ((np.sum(task_ids == task_preds) / len(task_ids)) * 100))
    
    return accuracy

def multitest(work_dir, pkl_path, log_file, json_name='result.json'):
    """
    Test all models in the work_dir
    """
    epochs = [int(pth_file.split('_')[-1].split('.')[0]) for pth_file in glob.glob(os.path.join(work_dir, '*.pth'))]
    accuracies = [test (work_dir, pkl_path, epoch, silent=True) for epoch in epochs]

    result_json = { k:{'task_acc': v} for (k, v) in zip(epochs, accuracies)}

    log_file = os.path.join(work_dir, log_file)
    int_regex = r"\d+"
    float_regex = r"[-+]?\d*\.\d+|\d+"
    for line in open(log_file, 'r'):
        epoch = int(re.findall(r"Epoch: +(\d+),", line)[0])
        lr = float(re.findall(r"lr: +([-+]?\d*\.\d+|\d+),", line)[0])
        train_loss = float(re.findall(r"Train Loss: +([-+]?\d*\.\d+|\d+),", line)[0])
        val_loss = re.findall(r"Val Loss: +([-+]?\d*\.\d+|\d+)", line)
        val_loss = float(val_loss[0]) if len(val_loss) > 0 else 'NA'

        if epoch in result_json:
            result_json[epoch]['lr'] = lr
            result_json[epoch]['train_loss'] = train_loss
            result_json[epoch]['val_loss'] = val_loss
    
    with open(os.path.join(work_dir, json_name), 'w') as outfile:
        json.dump(result_json, outfile, indent=4, sort_keys=True)

if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'combine':
        combine(args.pkl, args.tag, args.save)
    elif args.mode == 'nms':
        nms(args.in_pkl, args.out_pkl, args.thres, args.no_reg)
    elif args.mode == 'train':
        train(args.work_dir, args.pkl, args.validate)
    elif args.mode == 'test':
        test(args.work_dir, args.pkl, args.load)
    elif args.mode == 'multitest':
        multitest(args.work_dir, args.pkl, args.log_file)
    else:
        raise ValueError("Go Away")
