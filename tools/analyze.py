"""
Evaluate over every 5 epochs
"""
import os, re, json
import pickle
import subprocess, shlex
import argparse
import glob
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import termtables as tt


def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate various models")
    subparsers = parser.add_subparsers(help='Mode of operation', dest="mode")
    # parser.add_argument('mode', choices=['test', 'eval', 'TC', 'parse', 'plot'], help="Mode of operation")

    # For test
    parser_test = subparsers.add_parser('test', help='Run the trained models on test set')
    parser_test.add_argument('model_dir', type=str, help="Path of models directory")
    parser_test.add_argument('result_dir', type=str, help="Path of results directory")
    parser_test.add_argument('gpus', type=int, help="Number of GPUs to use")
    parser_test.add_argument('--start', type=int, help="Start epoch", default=10)
    parser_test.add_argument('--step', type=int, help="Step size of evaluation", default=5)

    # For pruning the proposals -- throw away too long / short proposals
    parser_prune = subparsers.add_parser('prune', help="Prune the proposals")
    parser_prune.add_argument('result_dir', type=str, help="Path of results directory")
    parser_prune.add_argument('prune_dir', type=str, help="Path of pruned results directory")
    parser_prune.add_argument('--l', type=float, default=0.05, help="Low Range")
    parser_prune.add_argument('--h', type=float, default=0.6, help="Hi range")    

    # For TC
    parser_tc = subparsers.add_parser('tc', help='Run COIN\'s TC on generated pickle files')
    parser_tc.add_argument('result_dir', type=str, help="Path of results directory")
    parser_tc.add_argument('result_tc_dir', type=str, help="Path of result TC directory")
    parser_tc.add_argument('--pooling', type=str, choices=['mean', 'max'], help="Type of pooling for TC", default='mean')

    # For TC--pruning
    parser_mtl_tc = subparsers.add_parser('mtl_tc', help='Run COIN\'s TC on generated pickle files using predicted tasks via MTL')
    parser_mtl_tc.add_argument('result_dir', type=str, help="Path of results directory")
    parser_mtl_tc.add_argument('result_tc_dir', type=str, help="Path of result TC directory")

    # For eval
    parser_eval = subparsers.add_parser('eval', help='Evaluate the pickle files for task accuracy and MAP scores')
    parser_eval.add_argument('model_dir', type=str, help="Path of models directory")
    parser_eval.add_argument('result_dir', type=str, help="Path of results directory")
    parser_eval.add_argument('eval_dir', type=str, help="Path of eval logs directory")

    # For parse
    parser_parse = subparsers.add_parser('parse', help='Parse the evaluated log files to extract the scores')
    parser_parse.add_argument('eval_dir', type=str, help="Path of eval logs directory")

    # For plot
    parser_plot = subparsers.add_parser('plot', help='Plot the parsed scores')
    parser_plot.add_argument('plot_type', choices=['task_score', 'map_0.1'], help="Type of plot")
    parser_plot.add_argument('--eval_dirs', type=str, nargs='+', help='List of evaluate directories', required=True)
    parser_plot.add_argument('--labels', type=str, nargs='+', help='List of plot labels', required=True)
    parser_plot.add_argument('--lo', type=float, required=True, help='Lower Y Limit')
    parser_plot.add_argument('--hi', type=float, required=True, help='Higher Y Limit')
    parser_plot.add_argument('--save_path', type=str, help='Path where to save the plot', required=True)
    parser_plot.add_argument('--title', type=str, default='', help='Plot title')
    
    # Transfering the weights of separately trained model to the big model
    parser_bootstrap = subparsers.add_parser('bs', help='Transfer the trained weights of different models')
    parser_bootstrap.add_argument('bs_mode', choices=['mtlssn', 'cons'])
    parser_bootstrap.add_argument('--ssn', type=str, help='Path of the SSN model')
    parser_bootstrap.add_argument('--mtlssn', type=str, help='Path of the MTL SSN model')
    parser_bootstrap.add_argument('--task_head', type=str, help='Path of the Task Head model')
    parser_bootstrap.add_argument('--cons_arch', type=str, help='Path of the cons_arch model')
    parser_bootstrap.add_argument('--save', type=str, help='Path where to save the transferred model')

    # Testing on final test set
    parser_test_bm = subparsers.add_parser('finaltest', help="Testing final best models on actual test set")
    parser_test_bm.add_argument('eval_path', type=str, help='Path of the eval python file')

    # Validating the args
    args = parser.parse_args()

    return args


def test(model_dir, result_dir, gpus, start=10, end=1000, step=5):
    """
    Test the different models and generate pickle files
    """
    try:
        os.makedirs(result_dir) # Try creating the directories
    except:
        pass

    for e in range(start, end + 1, step):
        model = os.path.join(model_dir, 'epoch_{}.pth'.format(e))
        if not os.path.exists(model):
            break # We are done

        config_file = os.path.join(model_dir, 'config.py')
        result = os.path.join(result_dir, 'result_{}.pkl'.format(e))
        log_file = os.path.join(result_dir, 'result_{}.log'.format(e))

        command = "python3 tools/test_localizer.py {} {} --gpus {} --out {}".format(
            config_file, model, gpus, result)
        print (command, '\n')
        command = shlex.split(command)
        with open(log_file, 'w') as outfile:
            process = subprocess.Popen(command, stdout=outfile)
        process.wait()


def prune(result_dir, prune_dir, low, hi):
    """
    Prune the pickle scores, by removing too large / short proposals
    """
    assert (result_dir != prune_dir)
    try:
        os.makedirs(prune_dir)
    except:
        pass

    pkl_files = glob.glob(os.path.join(result_dir, '*.pkl'))
    for pkl in pkl_files:
        file_name = pkl.split('/')[-1]
        prune_file = os.path.join(prune_dir, file_name)
        
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

        pickle.dump(results, open(prune_file, 'wb'))


def tc (result_dir, result_tc_dir, pooling='mean'):
    """
    Enforce task consistency (COIN's TC) over the generated scores
    """
    assert result_dir != result_tc_dir
    try:
        os.makedirs(result_tc_dir)
    except:
        pass

    pkl_files = glob.glob(os.path.join(result_dir, '*.pkl'))
    for pkl in pkl_files:
        file_name = pkl.split('/')[-1]
        tc_file = os.path.join(result_tc_dir, file_name)
        
        command = "python3 tools/localize_TC.py {} --out_pkl {} --pooling {}".format(
            pkl, tc_file, pooling)
        print (command, '\n')
        command = shlex.split(command)
        process = subprocess.Popen(command)


def mtl_tc (result_dir, result_tc_dir):
    """
    Enforce task consistency (using the MTL task predictions) over the generated scores
    """
    assert result_dir != result_tc_dir
    try:
        os.makedirs(result_tc_dir)
    except:
        pass

    pkl_files = glob.glob(os.path.join(result_dir, '*.pkl'))
    for pkl in pkl_files:
        file_name = pkl.split('/')[-1]
        tc_file = os.path.join(result_tc_dir, file_name)
        
        command = "python3 tools/localize_TC.py {} --out_pkl {} --mtl".format(
            pkl, tc_file)
        print (command, '\n')
        command = shlex.split(command)
        process = subprocess.Popen(command)


def evaluate (model_dir, result_dir, eval_dir):
    """
    Evaluate the generated pickle file
    """
    try:
        os.makedirs(eval_dir)
    except:
        pass

    config_file = os.path.join(model_dir, 'config.py')
    pkl_files = glob.glob(os.path.join(result_dir, '*.pkl'))
    for pkl in pkl_files:
        file_name = pkl.split('/')[-1].split('.')[0]
        log_file = os.path.join(eval_dir, '{}.log'.format(file_name))
        command = "python3 tools/eval_localize_results.py {} {} --eval coin".format(
            config_file, pkl)
        print (command, '\n')
        command = shlex.split(command)
        with open(log_file, 'w') as outfile:
            process = subprocess.Popen(command, stdout=outfile)
        time.sleep(10)


def parse_scores (eval_dir):
    """
    Parse log files and extract task accuracy and map scores
    """
    log_files = glob.glob(os.path.join(eval_dir, '*.log'))
    scores = dict()
    for log_file in tqdm(log_files):
        for line in open(log_file, 'r'):
            if re.search("Task Classification Accuracy:", line):
                task_acc_line = line
            if re.search("mean AP", line):
                map_line = line

        float_regex = r"[-+]?\d*\.\d+|\d+"
        task_acc = float(re.findall(float_regex, task_acc_line)[0])
        map_score = float(re.findall(float_regex, map_line)[0])

        int_regex = r"\d+"
        epoch_no = int(re.findall(int_regex, log_file)[-1])

        scores[epoch_no] = {"task_score": task_acc, "map_0.1": map_score}
    
    with open(os.path.join(eval_dir, 'scores.json'), 'w') as outfile:
        json.dump(scores, outfile, indent=4, sort_keys=True)


def plot (eval_dirs, labels, plot_type, save_path, low_limit, hi_limit, title=''):
    """
    Plot the task accuracy and map distributions
    """
    assert len(eval_dirs) == len(labels)

    fig = plt.figure()
    for eval_dir, label in zip(eval_dirs, labels):
        score_file = os.path.join(eval_dir, 'scores.json')
        with open(score_file) as f:
            scores = json.load(f)

        X = []; Y = []
        for epoch, score in scores.items():
            X.append(int(epoch))
            Y.append(float(score[plot_type]))

        plt.plot(X, Y, linestyle='solid', label=label)

    for x in range(100):
        plt.axhline(y=x * 0.01, linestyle=':', alpha=0.1, color='black') # Some axis
    for x in range(20):
        plt.axhline(y=5 * x * 0.01, linestyle='--', alpha=0.2, color='black') # Some axis

    plt.ylim(top=hi_limit)
    plt.ylim(bottom=low_limit)
    plt.legend()
    plt.title(title)
    plt.xlabel('# Epochs')
    plt.savefig(save_path)


def bootstrap(mode, ssn=None, mtlssn=None, cons_arch=None, task_head=None, epoch=100, save_checkpoint_pth='bootstrapped.pth'):
    assert mode in ['mtlssn', 'cons']
    assert ssn is not None and task_head is not None
    if mode == 'mtlssn': assert mtlssn is not None
    if mode == 'cons': assert cons_arch is not None

    models = dict()
    # Read the task and ssn weights
    def load_weights(checkpoint_pth, state_dict=True, get_meta_info=False):
        if torch.cuda.is_available():
            model = torch.load(checkpoint_pth)
        else:
            model = torch.load(checkpoint_pth, map_location=torch.device('cpu'))
        
        if get_meta_info:
            # Return info about the meta
            return model['meta']

        if state_dict:
            return model['state_dict']
        else:
            return model

    def get_top_keys(model):
        return set({w.split('.')[0] for w in model.keys()})

    def save_model(model_state_dict, meta_info):
        meta_info = {
            'epoch': epoch,
            'iter': epoch * meta_info['iter'] / meta_info['epoch']
        }
        torch.save(dict({
            'state_dict': model_state_dict,
            'meta': meta_info
        }), save_checkpoint_pth)

    models['ssn'] = load_weights(ssn)
    models['task_head'] = load_weights(task_head, state_dict=False)

    assert get_top_keys(models['ssn']) == set({'backbone', 'cls_head'})
    assert get_top_keys(models['task_head']) == set({'fcs'})

    if mode == 'mtlssn':
        # Transfer weights from task_head, SSN -> MTL arch
        template = load_weights(mtlssn)
        assert get_top_keys(template) == set({'backbone', 'cls_head', 'task_head'})
        for k, w in models['ssn'].items():
            template[k] = w
        for k, w in models['task_head'].items():
            template['task_head.%s' % k] = w
        meta_info = load_weights(mtlssn, get_meta_info=True)
        save_model(template, meta_info)
        # torch.save(dict({'state_dict': template}), save_checkpoint_pth)

    elif mode == 'cons':
        # Transfer weights from task,ssn -> cons
        template = load_weights(cons_arch)
        assert get_top_keys(template) == set({'backbone', 'cls_head', 'aux_task_head', 'task_head'})
        for k, w in models['ssn'].items():
            # Transfer only the backbone weights
            if k.split('.')[0] == 'backbone': template[k] = w
        for k, w in models['task_head'].items():
            template['task_head.%s' % k] = w
        meta_info = load_weights(cons_arch, get_meta_info=True)
        save_model(template, meta_info)
        # torch.save(dict({
        #     'state_dict': template,
        #     'meta': dict({
        #         'epoch': 100,
        #         'iter': 
        #     })
        # }), save_checkpoint_pth)

    else:
        raise ValueError("Invalid Mode")


def test_best_models(eval_path, prune_low_range=0.05, prune_high_range=0.6):
    """
    This function is used to get final test scores using the best val model
    """
    import sys
    sys.path.append(eval_path)
    from eval import models, refresh, result_dir, gpus
    
    for name in refresh:
        print ('Testing model %s...' % name)
        rdir = os.path.join(result_dir, name)
        try: os.makedirs(rdir)
        except: pass

        model = models[name]
        checkpoint = model['checkpoint']

        # Result Raw
        config_file = os.path.join(
            '/'.join(checkpoint.split('/')[:-1]),
            'config.py'
        )
        result_file = os.path.join(rdir, 'result_raw.pkl')
        log_file = os.path.join(rdir, 'test.log')

        command = "python3 tools/test_localizer.py {} {} --gpus {} --out {}".format(
            config_file, checkpoint, gpus, result_file)
        print (command, '\n')
        command = shlex.split(command)
        with open(log_file, 'w') as outfile:
            process = subprocess.Popen(command, stdout=outfile)
        process.wait()

        if 'tag_pruning' in model and model['tag_pruning']:
            # TAG Pruning
            result_pr_file = os.path.join(rdir, 'result_pr.pkl')
            pkl_data = pickle.load(open(result_file, 'rb'))

            results = []
            for data_idx, (props, act_scores, comp_scores, regs, task_scores) in enumerate(pkl_data):
                keep = []
                for idx, p in enumerate(props):
                    if (prune_low_range < p[1] - p[0] < prune_high_range):
                        keep.append(idx)
                if len(keep) == 0:
                    keep = [0] # Keep the first element
                    print ('index {} is completely useless!!'.format(data_idx))
                results.append((props[keep, :], act_scores[keep, :], comp_scores[keep, :], regs[keep, :, :], task_scores))

            pickle.dump(results, open(result_pr_file, 'wb'))
            result_file = result_pr_file

        # Consistency Pruning
        if 'TC' in model and model['TC'] == 'COIN':
            result_tc_file = os.path.join(rdir, 'result_tc.pkl')
            command = "python3 tools/localize_TC.py {} --out_pkl {} --pooling {}".format(
                result_file, result_tc_file, 'mean')
            print (command, '\n')
            command = shlex.split(command)
            process = subprocess.Popen(command)
            process.wait()
            result_file = result_tc_file
        elif 'TC' in model and model['TC'] == 'MTL':
            result_tc_file = os.path.join(rdir, 'result_tc.pkl')
            command = "python3 tools/localize_TC.py {} --out_pkl {} --mtl".format(
                result_file, result_tc_file)
            print (command, '\n')
            command = shlex.split(command)
            process = subprocess.Popen(command)
            process.wait()
            result_file = result_tc_file

        # Eval
        log_file = os.path.join(rdir, 'eval.log')
        command = "python3 tools/eval_localize_results.py {} {} --eval coin".format(
            config_file, result_file)
        print (command, '\n')
        command = shlex.split(command)
        with open(log_file, 'w') as outfile:
            process = subprocess.Popen(command, stdout=outfile)
        process.wait()

        print ('\n==================================\n')


    # Parse log files and extract task accuracy and map scores
    test_scores = []
    for name in models.keys():
        log_file = os.path.join(result_dir, name, 'eval.log')
        if not os.path.isfile(log_file):
            continue

        for line in open(log_file, 'r'):
            if re.search("Task Classification Accuracy:", line):
                task_acc_line = line
            if re.search("mean AP", line):
                map_line = line

        float_regex = r"[-+]?\d*\.\d+|\d+"
        task_acc = float(re.findall(float_regex, task_acc_line)[0])
        map_scores = [float(s) for s in re.findall(float_regex, map_line)]

        scores = [name]+map_scores+[task_acc]
        test_scores.append(scores)

    string = tt.to_string(
        test_scores,
        header=["model"]+['mAP @ 0.%d' % i for i in range(1, 10)]+["Task Acc."],
        style=tt.styles.ascii_thin_double,
        padding=(0, 1),
        alignment="c"*11
    )
    print (string)

if __name__ == '__main__':
    # time.sleep(1)
    args = parse_args()

    if args.mode == 'prune':
        assert 0 < args.l < args.h < 1
        prune(args.result_dir, args.prune_dir, args.l, args.h)
    elif args.mode == 'test':
        test(args.model_dir, args.result_dir, args.gpus, start=args.start, step=args.step)
    elif args.mode == 'tc':
        tc (args.result_dir, args.result_tc_dir, args.pooling)
    elif args.mode == 'mtl_tc':
        mtl_tc (args.result_dir, args.result_tc_dir)
    elif args.mode == 'eval':
        evaluate (args.model_dir, args.result_dir, args.eval_dir)
    elif args.mode == 'parse':
        parse_scores (args.eval_dir)
    elif args.mode == 'plot':
        plot (args.eval_dirs, args.labels, args.plot_type, args.save_path, args.lo, args.hi, args.title)
    elif args.mode == 'bs':
        bootstrap(args.bs_mode, ssn=args.ssn, mtlssn=args.mtlssn, cons_arch=args.cons_arch, task_head=args.task_head, save_checkpoint_pth=args.save)
    elif args.mode == 'finaltest':
        test_best_models(args.eval_path)
    else:
        raise ValueError("Go Away")
