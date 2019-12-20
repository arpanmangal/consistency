"""
Evaluate over every 5 epochs
"""
import os, re, json
import subprocess, shlex
import argparse
import glob
import time
import matplotlib.pyplot as plt


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

    # For TC
    parser_tc = subparsers.add_parser('tc', help='Run COIN\'s TC on generated pickle files')
    parser_tc.add_argument('result_dir', type=str, help="Path of results directory")
    parser_tc.add_argument('result_tc_dir', type=str, help="Path of result TC directory")
    parser_tc.add_argument('--pooling', type=str, choices=['mean', 'max'], help="Type of pooling for TC", default='mean')

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
    parser_plot.add_argument('--save_path', type=str, help='Path where to save the plot', required=True)
    parser_plot.add_argument('--title', type=str, default='', help='Plot title')
    
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


def tc (result_dir, result_tc_dir, pooling='mean'):
    """
    Enforce term consistency over the generated scores
    """
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
        process.wait()
        

def parse_scores (eval_dir):
    """
    Parse log files and extract task accuracy and map scores
    """
    log_files = glob.glob(os.path.join(eval_dir, '*.log'))
    scores = dict()
    for log_file in log_files:
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


def plot (eval_dirs, labels, plot_type, save_path, title=''):
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

    plt.legend()
    plt.title(title)
    plt.xlabel('# Epochs')
    plt.savefig(save_path)


if __name__ == '__main__':
    # time.sleep(1)
    args = parse_args()
    if args.mode == 'test':
        test(args.model_dir, args.result_dir, args.gpus, start=args.start, step=args.step)
    elif args.mode == 'TC':
        tc (args.result_dir, args.result_tc_dir, args.pooling)
    elif args.mode == 'eval':
        evaluate (args.model_dir, args.result_dir, args.eval_dir)
    elif args.mode == 'parse':
        parse_scores (args.eval_dir)
    elif args.mode == 'plot':
        plot (args.eval_dirs, args.labels, args.plot_type, args.save_path, args.title)
