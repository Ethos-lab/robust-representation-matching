import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn

from robustness.datasets import DATASETS
from robustness import cifar_models, imagenet_models
from utils import load_checkpoint, ModelwithInputNormalization

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescentPyTorch, AutoProjectedGradientDescent, AutoAttack


def get_args():
    parser = argparse.ArgumentParser(description="Testing script using IBM ART (defaults for cifar).")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"],
                        default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/CIFAR10", help="path to dataset if you need diff from default")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--load-path", type=str, default="checkpoint.pt.last")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams
    parser.add_argument("--attack", type=str, default="pgd", choices=['pgd', 'autopgd'])
    parser.add_argument("--eps", type=float, default=1.0, help="perturbation budget")
    parser.add_argument("--pgd-iters", type=int, default=50, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.125, help="perturb step size")
    parser.add_argument("--constraint", type=str, default="2", choices=["2", "inf"], help="PGD attack constraint metric")
    parser.add_argument("--rand-starts", type=int, default=10, help="random initialization for PGD")

    args = parser.parse_args()

    return args


def test(model, test_loader, attack=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader), leave=False)
    for im, lbl in test_loader:
        im, lbl = im.numpy(), lbl.numpy()
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
    #test_loader = helpers.DataPrefetcher(test_loader)

    if args.dataset == "cifar":
        nb_classes = 10
        input_shape = (3,32,32)
        model = cifar_models.__dict__[args.arch](num_classes=nb_classes)
    else:
        nb_classes = 9
        input_shape = (3,224,224)
        model = imagenet_models.__dict__[args.arch](num_classes=nb_classes)

    # Loading model checkpoint
    ckpt = load_checkpoint(args.load_path)
    model.load_state_dict(ckpt['model_sd'])
    print(f"Successfully loaded checkpoint from epoch: {ckpt['load_epoch']} !!!")
    del ckpt

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = ModelwithInputNormalization(model, dataset.mean, dataset.std)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    art_model = PyTorchClassifier(model=model, loss=criterion, input_shape=input_shape,
                                  nb_classes=nb_classes, clip_values=(0,1))


    # Testing begins

    # Natural accuracy
    acc = test(art_model, test_loader)
    print(f"Natural accuracy: {acc:.4f}%")

    try:
        norm = int(args.constraint)
    except ValueError:
        if args.constraint == 'inf': norm = np.inf
        else: raise ValueError

    if args.attack == 'pgd':
       attack_kwargs = {
           "norm": norm,
           "eps": args.eps,
           "eps_step": args.step_size,
           "max_iter": args.pgd_iters,
           "targeted": False,
           "num_random_init": args.rand_starts,
           "batch_size": args.batch_size,
           "verbose": False
       }
       adversary = ProjectedGradientDescentPyTorch(art_model, **attack_kwargs)
    else:
        adversary = AutoProjectedGradientDescent(art_model, norm=norm, eps=args.eps, eps_step=2 * args.eps,
                                                 max_iter=args.pgd_iters, targeted=False, batch_size=args.batch_size,
                                                 verbose=False, nb_random_init=args.rand_starts)

    acc = test(art_model, test_loader, adversary)
    print(f"Adversarial accuracy ({args.eps}): {acc:.4f}%")


if __name__ == "__main__":
    main()

