"""
The main file, which exposes the robustness command-line tool, detailed in
:doc:`this walkthrough <../example_usage/cli_usage>`.
"""

from argparse import ArgumentParser
import os
import git
import torch as ch
from torch.utils.data import DataLoader

import cox
import cox.utils
import cox.store

try:
    from .model_utils import make_and_restore_model
    from .datasets import DATASETS
    from .train import train_model, eval_model
    from .tools import constants, helpers, folder
    from . import defaults, __version__
    from .defaults import check_and_fill_args
    from .data_augmentation import TRAIN_TRANSFORMS_DEFAULT


except:
    raise ValueError("Make sure to run with python -m (see README.md)")


parser = ArgumentParser()
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

def main(args, store=None):
    '''Given arguments from `setup_args` and a store from `setup_store`,
    trains as a model. Check out the argparse object in this file for
    argument options.
    '''

    # MAKE DATASET AND LOADERS
    data_path = os.path.expandvars(args.data)
    dataset = DATASETS[args.dataset](data_path)

#    train_loader, val_loader = dataset.make_loaders(args.workers,
#                    args.batch_size, data_aug=bool(args.data_aug))

    #############

    train_data = ch.cat(ch.load("/media/big_hdd/data/CIFAR10_constructed/d_robust_CIFAR/CIFAR_ims")).cpu()
    train_labels = ch.cat(ch.load("/media/big_hdd/data/CIFAR10_constructed/d_robust_CIFAR/CIFAR_lab")).cpu()
    train_set = folder.TensorDataset(train_data, train_labels,
                                     transform=TRAIN_TRANSFORMS_DEFAULT(32))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)

    #NOTE: Still need arg: dataroot to load val set.
    _, val_loader = dataset.make_loaders(only_val=True,
                                        batch_size=128,
                                        shuffle_val=False,
                                        workers=args.workers)


    ################

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)
    loaders = (train_loader, val_loader)

    print(args)
    parallel = False
    # Run on multiple GPUs
    if ch.cuda.device_count() > 1:
        parallel = True 
    # MAKE MODEL
    model, checkpoint = make_and_restore_model(arch=args.arch,
            dataset=dataset, resume_path=args.resume, parallel=parallel)
    if 'module' in dir(model): model = model.module

    if args.eval_only:
        return eval_model(args, model, val_loader, store=store)

    if not args.resume_optimizer: checkpoint = None
    model = train_model(args, model, loaders, store=store,
                                    checkpoint=checkpoint)
    return model

def setup_args(args):
    '''
    Fill the args object with reasonable defaults from
    :mod:`robustness.defaults`, and also perform a sanity check to make sure no
    args are missing.
    '''
    # override non-None values with optional config_path
    if args.config_path:
        args = cox.utils.override_json(args, args.config_path)

    ds_class = DATASETS[args.dataset]
    args = check_and_fill_args(args, defaults.CONFIG_ARGS, ds_class)

    if not args.eval_only:
        args = check_and_fill_args(args, defaults.TRAINING_ARGS, ds_class)

    if args.adv_train or args.adv_eval:
        args = check_and_fill_args(args, defaults.PGD_ARGS, ds_class)

    args = check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, ds_class)
    if args.eval_only: assert args.resume is not None, \
            "Must provide a resume path if only evaluating"
    return args

def setup_store_with_metadata(args):
    '''
    Sets up a store for training according to the arguments object. See the
    argparse object above for options.
    '''
    # Add git commit to args
    try:
        repo = git.Repo(path=os.path.dirname(os.path.realpath(__file__)),
                            search_parent_directories=True)
        version = repo.head.object.hexsha
    except git.exc.InvalidGitRepositoryError:
        version = __version__
    args.version = version

    # Create the store
    store = cox.store.Store(args.out_dir, args.exp_name)
    args_dict = args.__dict__
    schema = cox.store.schema_from_dict(args_dict)
    store.add_table('metadata', schema)
    store['metadata'].append_row(args_dict)
    return store

if __name__ == "__main__":
    args = parser.parse_args()
    args = cox.utils.Parameters(args.__dict__)

    args = setup_args(args)
    store = setup_store_with_metadata(args)
    #store = None
    final_model = main(args, store=store)
