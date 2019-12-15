"""
Evaluate over every 5 epochs
"""
import os
import subprocess, shlex
import argparse
import glob
import time

def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate various models")
    parser.add_argument('mode', choices=['test', 'eval', 'TC'], help="Mode of operation")

    # For test
    parser.add_argument('--model_dir', '-m', type=str, help="Path of models directory")
    parser.add_argument('--result_dir', '-r', type=str, help="Path of results directory")
    parser.add_argument('--gpus', type=int, help="Number of GPUs to use")
    parser.add_argument('--start', type=int, help="Start epoch", default=10)
    parser.add_argument('--step', type=int, help="Step size of evaluation", default=5)

    # For TC
    parser.add_argument('--result_tc_dir', '-tc', type=str, help="Path of result TC directory")
    parser.add_argument('--pooling', type=str, choices=['mean', 'max'], help="Type of pooling for TC", default='mean')

    # For eval
    parser.add_argument('--eval_dir', '-e', type=str, help="Path of eval logs directory")

    # Validating the args
    args = parser.parse_args()

    if args.mode == 'test':
        assert args.model_dir is not None
        assert args.result_dir is not None
        assert args.gpus is not None
    elif args.mode == 'TC':
        assert args.result_dir is not None
        assert args.result_tc_dir is not None
    else:
        assert args.model_dir is not None
        assert args.result_dir is not None
        assert args.eval_dir is not None

    # parser.add_argument('pkl', help='Path of the result pkl file')
    # parser.add_argument('--out_pkl', help='Path of the output pkl file')
    # parser.add_argument('--W', help='Path to the W matrix', default='data/coin/W.npy')
    # parser.add_argument('--pooling', help='How to pool. Should be either \'mean\' or \'max\'',
    #                     type=str, default='mean', choices=['mean', 'max'])
    
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

if __name__ == '__main__':
    time.sleep(1)
    args = parse_args()
    if args.mode == 'test':
        test(args.model_dir, args.result_dir, args.gpus, start=args.start, step=args.step)
    elif args.mode == 'TC':
        tc (args.result_dir, args.result_tc_dir, args.pooling)
    else:
        evaluate (args.model_dir, args.result_dir, args.eval_dir)

