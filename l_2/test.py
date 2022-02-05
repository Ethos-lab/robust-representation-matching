import tqdm
import argparse
import numpy as np

import torch as ch
import torch.nn as nn

from robustness import cifar_models, imagenet_models
from robustness.attacker import AttackerModel
from robustness.datasets import DATASETS
from robustness.tools import helpers
from utils import load_checkpoint

ch.manual_seed(0)

'''
NOTE
----
General rule of thumb:
  - step_size = 2.5*eps/iterations
'''


def get_args():
    parser = argparse.ArgumentParser(description="Testing script (defaults for cifar).")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"], default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="/media/big_hdd/data/CIFAR10", help="path to dataset")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--load-path", type=str, default="checkpoint.pt.last")
    parser.add_argument("--batch-size", type=int, default=50, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams
    parser.add_argument("--eps", type=float, default=1.0, help="perturbation budget")
    parser.add_argument("--pgd-iters", type=int, default=50, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.125, help="perturb step size")
    parser.add_argument("--constraint", type=str, default="2", choices=["2", "inf"], help="PGD attack constraint metric")
    parser.add_argument("--rand-starts", type=int, default=10, help="random initialization for PGD")

    args = parser.parse_args()

    return args


def test(amodel, test_loader, make_adv, attack_kwargs=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader), leave=False)
    mean_logits = np.zeros(10)
    for im, lbl in test_loader:
        im, lbl = im.cuda(), lbl.cuda()
        if make_adv:
            assert attack_kwargs != None
            _, im_adv = amodel(im, lbl, make_adv=True, **attack_kwargs)
            pred, _ = amodel(im_adv)
        else:
            pred, _ = amodel(im, make_adv=False)
        mean_logits += ch.mean(pred, dim=0).detach().cpu().numpy()
        label_pred = ch.argmax(pred, dim=1)
        correct += (label_pred == lbl).sum().item()
        total += im.shape[0]
        pbar.update(1)
    pbar.close()
#    print(mean_logits / len(test_loader))

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
    test_loader = helpers.DataPrefetcher(test_loader)

    if args.dataset == "cifar":
        model = cifar_models.__dict__[args.arch](num_classes=10)
    else:
        model = imagenet_models.__dict__[args.arch](num_classes=9)

    # Loading model checkpoint
    ckpt = load_checkpoint(args.load_path)
    model.load_state_dict(ckpt['model_sd'])
    print(f"Successfully loaded checkpoint from epoch: {ckpt['load_epoch']} !!!")
    del ckpt

    if ch.cuda.device_count() > 1:
        print("Let's use", ch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
 
    # wrap nn.module object into AttackerModel to make it compatible with robustness toolbox
    amodel = AttackerModel(model, dataset).cuda()
    amodel.eval()

    # Clean eval
    acc = test(amodel, test_loader, make_adv=False)
    print(f"Natural accuracy: {acc:.4f}%")

    # PGD eval
    attack_kwargs = {
        "constraint": args.constraint,
        "eps": args.eps,
        "step_size": args.step_size,
        "iterations": args.pgd_iters,
        "random_start": (args.rand_starts > 0),
        "random-restarts": args.rand_starts,
        "do_tqdm": False
    }
    acc = test(amodel, test_loader, make_adv=True, attack_kwargs=attack_kwargs)
    print(f"Adversarial accuracy (eps: {args.eps}): {acc:.4f}%")


if __name__ == "__main__":
    main()
