import os
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch, AutoProjectedGradientDescent, AutoAttack

try:
    from robustness import cifar_models
    from l_inf.utils import (cifar10_mean, cifar10_std, get_loaders, ModelwithInputNormalization, load_checkpoint)
except:
    raise ValueError("Make sure to run with python -m from root project directory")


def get_args():
    parser = argparse.ArgumentParser(description="Testing script using IBM ART attack implementations.")
    parser.add_argument("--dataroot", type=str, default="./CIFAR10",
                        help="path to dataset if you need diff from default")
    parser.add_argument("--arch", type=str, default="resnet50", choices=['resnet18', 'resnet50', 'vgg11', 'vgg19'])
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams (no default values, pass every time)
    parser.add_argument("--eps", type=float, default=0.03137254901960784, help="perturbation budget (default: 8/255)")
    parser.add_argument("--pgd-iters", type=int, default=20, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.00784313725490196,
                        help="perturb step size (default: 2/255)")
    parser.add_argument("--random-restarts", default=1, type=int, help="random initialization for PGD")
    parser.add_argument("--attack", type=str, default='pgd', choices=['pgd', 'autopgd'])

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

    return 100 * correct / total


def compute_time_stats(logfile):
    all_vals = []
    data = open(logfile, 'r').read().split('\n')[:-1]
    # Hack for identifying the logging format
    if 'Epoch Seconds LR Train Loss Train Acc' in ' '.join(data[1].split()):
        for d in data[2:]:
            if 'Total train time' in d:
                break
            tmp = d.split('\t')
            all_vals.append(float(tmp[1]))
    elif 'Train |' in data[1]:
        for d in data:
            if 'Epoch' in d:
                tmp = d.split('\t')
                all_vals.append(float(tmp[1].split(':')[-1].strip()))
    else:
        raise ValueError('Unable to identify log file format !!!')

    # Computing 95% confidence interval for average epoch time
    pop_size = len(all_vals)
    pop_mean = np.mean(all_vals)
    pop_std = np.std(all_vals)
    std_err = pop_std / np.sqrt(pop_size)
    ci = 1.962 * std_err    # 95% confidence interval

    if 'free_at' in logfile:
        tmp = data[0].split(',')
        for t in tmp:
            if 'minibatch_replays' in t: minibatch_replays = int(t.split('=')[-1])
        pop_mean /= minibatch_replays
        pop_size *= minibatch_replays

    return {'mean': pop_mean, 'ci': ci, 'epochs': pop_size}


def main():
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    # Get arguments
    args = get_args()
    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # Load dataset
    train_loader, test_loader = get_loaders(args.dataroot, args.batch_size, workers=args.num_workers)

    model = cifar_models.__dict__[args.arch](num_classes=10)
    model = ModelwithInputNormalization(model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))

    # Loading model checkpoint
    ckpt = load_checkpoint(args.load_path, 'eval', device)
    model.net.load_state_dict(ckpt['model_sd'])
    print(f"Successfully loaded checkpoint from epoch: {ckpt['load_epoch']} !!!")
    del ckpt

    model.float()
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    art_model = \
        PyTorchClassifier(model=model, loss=criterion, input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 1))

    # Training time statistics
    logfile = f'{args.load_path.replace(args.load_path.split("/")[-1], "output.log")}'
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
            "norm": np.inf,  # L_inf attack
            "eps": args.eps,
            "eps_step": args.step_size,
            "max_iter": args.pgd_iters,
            "targeted": False,
            "num_random_init": args.random_restarts,
            "batch_size": args.batch_size,
            "verbose": False
        }
        adversary = ProjectedGradientDescentPyTorch(art_model, **attack_kwargs)

    else:
        attack_kwargs = {
            "norm": np.inf,  # L_inf attack
            "eps": args.eps,
            "eps_step": 2 * args.eps,
            "max_iter": args.pgd_iters,
            "targeted": False,
            "nb_random_init": args.random_restarts,
            "batch_size": args.batch_size,
            "verbose": False
        }
        adversary = AutoProjectedGradientDescent(art_model, **attack_kwargs)

    acc = test(art_model, test_loader, adversary)
    print(f"Adversarial accuracy ({args.eps:.4f}): {acc:.4f}%")


if __name__ == "__main__":
    main()
