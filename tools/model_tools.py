"""
Script to transfer weights of different models to other models
"""

import argparse
import torch


def parse_args():
    """
    Create an argument parser and return parsed arguments
    """
    parser = argparse.ArgumentParser(description="Transfer weights of various models")
    subparsers = parser.add_subparsers(help='Mode of operation', dest="mode")

    # Transfer selective weights from one model to another
    parser_transfer = subparsers.add_parser('trans', help='Transfer selective weights from one model to another')
    parser_transfer.add_argument('base', type=str, help='Path of the base model')
    parser_transfer.add_argument('template', type=str, help='Path of the template to which transfer weights')
    parser_transfer.add_argument('save', type=str, help='Path where to save the template with base weights')
    parser_transfer.add_argument('--old_base', action='store_true', default=False, help='Whether the base is an old PyTorch model')
    parser_transfer.add_argument('--modules', type=str, nargs='+', help='Which modules to transfer weights of')

    # Transfering the weights of separately trained model to the big model
    parser_bootstrap = subparsers.add_parser('bs', help='Transfer the trained weights of different models')
    parser_bootstrap.add_argument('bs_mode', choices=['mtlssn', 'cons'])
    parser_bootstrap.add_argument('--ssn', type=str, help='Path of the SSN model')
    parser_bootstrap.add_argument('--mtlssn', type=str, help='Path of the MTL SSN model')
    parser_bootstrap.add_argument('--task_head', type=str, help='Path of the Task Head model')
    parser_bootstrap.add_argument('--cons_arch', type=str, help='Path of the cons_arch model')
    parser_bootstrap.add_argument('--save', type=str, help='Path where to save the transferred model')

    # Remove optimizer
    parser_ro = subparsers.add_parser('ro', help='Transfer the trained weights of different models')
    parser_ro.add_argument('-m', type=str, help='Path of the SSN model')
    parser_ro.add_argument('--save', type=str)

    # Transfer backbone weights
    parser_tb = subparsers.add_parser('tb', help='Transfer the trained weights of different models')
    parser_tb.add_argument('-a', type=str, help='Path of the SSN model')
    parser_tb.add_argument('-b', type=str, help='Path of the MTL SSN model')
    parser_tb.add_argument('--save', type=str)

    # Transfering weights from old backbone to newer backbone
    parser_oldweights = subparsers.add_parser('oback', help='Transfer the old backbone weights to new model')
    parser_oldweights.add_argument('--old', required=True, type=str, help='Path of the old model')
    parser_oldweights.add_argument('--new', required=True, type=str, help='Path of the new model')
    parser_oldweights.add_argument('--save', required=True, type=str, help='Path where to save the transferred model')

    # Testing on final test set
    parser_test_bm = subparsers.add_parser('finaltest', help="Testing final best models on actual test set")
    parser_test_bm.add_argument('eval_path', type=str, help='Path of the eval python file')

    # Validating the args
    args = parser.parse_args()

    return args


def load_model(checkpoint_pth):
    """Read the task and ssn weights"""
    if torch.cuda.is_available():
        model = torch.load(checkpoint_pth)
    else:
        model = torch.load(checkpoint_pth, map_location=torch.device('cpu'))
    return model


def load_weights(checkpoint_pth):
    """Load network weights"""
    model = load_model(checkpoint_pth)
    if 'state_dict' in model.keys():
        return model['state_dict']
    else:
        return model


def load_meta(checkpoint_pth):
    """Load meta info"""
    model = load_model(checkpoint_pth)
    assert 'meta' in model.keys()
    return model['meta']


def get_top_key(key):
    return key.split('.')[0]


def get_top_keys(state_dict):
    """Returns the top keys in the state_dict"""
    return set({get_top_key(k) for k in state_dict.keys()})


def save_model(state_dict, meta_info, save_checkpoint_pth, epoch=1):
    """save the modified model checkpoint"""
    meta_info = {
        'epoch': epoch,
        'iter': epoch * meta_info['iter'] / meta_info['epoch']
    }
    torch.save(dict({
        'state_dict': state_dict,
        'meta': meta_info
    }), save_checkpoint_pth)


def transfer_weights(base, template, save_checkpoint_pth, modules, old_base=False):
    """
    Transfer weights from base to template
    """
    if old_base:
        raise NotImplementedError

    base_state_dict = load_weights(base)
    base_meta = load_meta(base)
    template_state_dict = load_weights(template)
    modules = set(modules)

    # Each module to transfer should be present in the base model as well as the template
    base_keys = get_top_keys(base_state_dict)
    template_keys = get_top_keys(template_state_dict)
    for key in modules:
        assert key in base_keys
        assert key in template_keys

    for key, weight in base_state_dict.items():
        if get_top_key(key) in modules:
            template_state_dict[key] = weight

    save_model(template_state_dict, base_meta, save_checkpoint_pth)

    
def remove_optimizer(base, save_checkpoint_pth):
    """
    Remove the optimizer object from base
    """
    base_state_dict = load_weights(base)
    base_meta = load_meta(base)
    save_model(base_state_dict, base_meta, save_checkpoint_pth, epoch=base_meta['epoch'])


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'trans':
        transfer_weights(args.base, args.template, args.save, args.modules, args.old_base)
    else:
        raise ValueError("INVALID MODE")