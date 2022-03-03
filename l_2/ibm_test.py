import os.path
import sys
sys.path.append('../')

import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch, AutoProjectedGradientDescent, AutoAttack

try:
    from robustness.tools import helpers
    from robustness.datasets import DATASETS
    from robustness import cifar_models, imagenet_models
    from l_2.utils import load_checkpoint, ModelwithInputNormalization
except:
    raise ValueError("Make sure to run with python -m from root project directory")


def get_args():
    parser = argparse.ArgumentParser(description="Testing script using IBM ART (defaults for cifar).")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"],
                        default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="./CIFAR10", help="path to dataset if you need diff from default")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--load-path", type=str, default="/path/to/checkpoint.pt")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature scaling for logits")
    # attack hyperparams
    parser.add_argument("--attack", type=str, default="pgd", choices=['pgd', 'autopgd'])
    parser.add_argument("--eps", type=float, default=1.0, help="perturbation budget")
    parser.add_argument("--pgd-iters", type=int, default=50, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.125, help="perturb step size")
    parser.add_argument("--random-restarts", type=int, default=10, help="random initialization for PGD")

    args = parser.parse_args()

    return args


def test(model, test_loader, attack=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader), leave=False)
    for im, lbl in test_loader:
        im, lbl = im.to('cpu').numpy(), lbl.to('cpu').numpy()
        if attack:
            im_adv = attack.generate(im, lbl)
            pred = model.predict(im_adv)
        else:
            pred = model.predict(im)
        label_pred = np.argmax(pred, axis=1)
        correct += (label_pred == lbl).sum()
        total += im.shape[0]
        pbar.update(1)
    pbar.close()

    return 100*correct/total


def compute_time_stats(logfile):
    all_vals = []
    data = open(logfile, 'r').read().split('\n')[:-1]
    # Hack for identifying the logging format
    if 'Argument' in data[0]:
        for d in data:
            if 'Train |' in d:
                tmp = d.split('|')[2]
                assert 'time taken' in tmp, "Failed to parse log file  !!!"
                all_vals.append(float(tmp.split(':')[-1].strip()[:-1]))
    else:
        raise ValueError('Unable to identify log file format !!!')

    # Computing 95% confidence interval for average epoch time
    pop_size = len(all_vals)
    pop_mean = np.mean(all_vals)
    pop_std = np.std(all_vals)
    std_err = pop_std / np.sqrt(pop_size)
    ci = 1.962 * std_err    # 95% confidence interval

    return {'mean': pop_mean, 'ci': ci, 'epochs': pop_size}


def main():
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    # Get arguments
    args = get_args()

    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # Load dataset
    dataset_function = DATASETS[args.dataset]
    dataset = dataset_function(args.dataroot)
    _, test_loader = dataset.make_loaders(only_val=True,
                                          batch_size=args.batch_size,
                                          shuffle_val=False,
                                          workers=args.num_workers)
    if device == 'cuda:0':
        test_loader = helpers.DataPrefetcher(test_loader)

    if args.dataset == "cifar":
        nb_classes = 10
        input_shape = (3,32,32)
        model = cifar_models.__dict__[args.arch](num_classes=nb_classes, temperature=args.temperature)
    else:
        nb_classes = 9
        input_shape = (3,224,224)
        model = imagenet_models.__dict__[args.arch](num_classes=nb_classes)

    # Loading model checkpoint
    ckpt = load_checkpoint(args.load_path, 'eval', device)
    model.load_state_dict(ckpt['model_sd'])
    print(f"Successfully loaded checkpoint from epoch: {ckpt['load_epoch']} !!!")
    del ckpt

    model = ModelwithInputNormalization(model, dataset.mean, dataset.std)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    art_model = PyTorchClassifier(model=model, loss=criterion, input_shape=input_shape,
                                  nb_classes=nb_classes, clip_values=(0,1))

    # Training time statistics
    logfile = f'{args.load_path.replace(args.load_path.split("/")[-1], "log.txt")}'
    if not os.path.exists(logfile):
        print("Unable to find log file, skipping train time stats computation !!!")
    else:
        res = compute_time_stats(logfile)
        print(f'Average epoch time: {res["mean"]:.2f}s, 95% confidence interval: {res["ci"]:.2f}s')
        print(f'Total training time: {res["mean"] * res["epochs"]/3600:.2f}h or {res["mean"] * res["epochs"]/60:.2f}m or {res["mean"] * res["epochs"]:.2f}s ')

    acc = test(art_model, test_loader)
    print(f"Clean accuracy: {acc:.4f}%")

    if args.attack == 'pgd':
       attack_kwargs = {
           "norm": 2,
           "eps": args.eps,
           "eps_step": args.step_size,
           "max_iter": args.pgd_iters,
           "targeted": False,
           "batch_size": args.batch_size,
           "num_random_init": args.random_restarts,
           "verbose": False
       }
       adversary = ProjectedGradientDescentPyTorch(art_model, **attack_kwargs)
    else:
        attack_kwargs = {
            "norm": 2,
            "eps": args.eps,
            "eps_step": 2 * args.eps,
            "max_iter": args.pgd_iters,
            "targeted": False,
            "batch_size": args.batch_size,
            "nb_random_init": args.random_restarts,
            "verbose": False
        }
        adversary = AutoProjectedGradientDescent(art_model, **attack_kwargs)

    acc = test(art_model, test_loader, adversary)
    print(f"Adversarial accuracy ({args.eps}): {acc:.4f}%")


if __name__ == "__main__":
    main()

