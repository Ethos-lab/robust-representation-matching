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
    from robustness import cifar_models
    from l_inf.utils import (cifar10_mean, cifar10_std, get_loaders, ModelwithInputNormalization, load_checkpoint)
except:
    raise ValueError("Make sure to run with python -m from root project directory")


def get_args():
    parser = argparse.ArgumentParser(description="Testing script using IBM ART attack implementations.")
    parser.add_argument("--dataroot", type=str, default="./CIFAR10", help="path to dataset if you need diff from default")
    parser.add_argument("--arch", type=str, default="resnet50", choices=['resnet18', 'resnet50', 'vgg11', 'vgg19'])
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams (no default values, pass every time)
    parser.add_argument("--eps", type=float, default=0.03137254901960784, help="perturbation budget (default: 8/255)")
    parser.add_argument("--pgd-iters", type=int, default=20, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.00784313725490196, help="perturb step size (default: 2/255)")
    parser.add_argument("--random-restarts", default=1, type=int, help="random initialization for PGD")
    parser.add_argument("--attack", type=str, default='pgd', choices=['pgd', 'auto_pgd'])

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
    art_model =\
        PyTorchClassifier(model=model, loss=criterion, input_shape=(3, 32, 32), nb_classes=10, clip_values=(0, 1))

    # Testing begins

    # Natural accuracy
    acc = test(art_model, test_loader)
    print(f"Natural accuracy: {acc:.4f}%")

    if args.attack == 'pgd':
        attack_kwargs = {
            "norm": np.inf, # L_inf attack
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

