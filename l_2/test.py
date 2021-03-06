import os
import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn

try:
    from robustness import cifar_models, imagenet_models
    from robustness.attacker import AttackerModel
    from robustness.datasets import DATASETS
    from robustness.tools import helpers
    from l_2.utils import load_checkpoint
except:
    raise ValueError("Make sure to run with python -m from root project directory")

torch.manual_seed(0)

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
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature scaling for logits")
    # attack hyperparams
    parser.add_argument("--eps", type=float, default=1.0, help="perturbation budget")
    parser.add_argument("--pgd-iters", type=int, default=50, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.125, help="perturb step size")
    parser.add_argument("--constraint", type=str, default="2", choices=["2", "inf"], help="PGD attack constraint metric")
    parser.add_argument("--random-restarts", type=int, default=10, help="random initialization for PGD")

    args = parser.parse_args()

    return args


def test(amodel, test_loader, device, make_adv, attack_kwargs=None):
    correct = 0
    total = 0
    pbar = tqdm.tqdm(total=len(test_loader), leave=False)
    for im, lbl in test_loader:
        im, lbl = im.to(device), lbl.to(device)
        if make_adv:
            assert attack_kwargs != None
            _, im_adv = amodel(im, lbl, make_adv=True, **attack_kwargs)
            pred, _ = amodel(im_adv)
        else:
            pred, _ = amodel(im, make_adv=False)
        label_pred = torch.argmax(pred, dim=1)
        correct += (label_pred == lbl).sum().item()
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
    if torch.cuda.device_count() > 0:
        test_loader = helpers.DataPrefetcher(test_loader)

    if args.dataset == "cifar":
        model = cifar_models.__dict__[args.arch](num_classes=10, temperature=args.temperature)
    else:
        model = imagenet_models.__dict__[args.arch](num_classes=9)

    # Loading model checkpoint
    ckpt = load_checkpoint(args.load_path, mode='eval', device=device)
    model.load_state_dict(ckpt['model_sd'])
    print(f"Successfully loaded checkpoint from epoch: {ckpt['load_epoch']} !!!")
    del ckpt

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
 
    # wrap nn.module object into AttackerModel to make it compatible with robustness toolbox
    amodel = AttackerModel(model, dataset).to(device)
    amodel.eval()

    # Training time statistics
    logfile = f'{args.load_path.replace(args.load_path.split("/")[-1], "log.txt")}'
    if not os.path.exists(logfile):
        print("Unable to find log file, skipping train time stats computation !!!")
    else:
        res = compute_time_stats(logfile)
        print(f'Average epoch time: {res["mean"]:.2f}s, 95% confidence interval: {res["ci"]:.2f}s')
        print(f'Total training time: {res["mean"] * res["epochs"]/3600:.2f}h or {res["mean"] * res["epochs"]/60:.2f}m or {res["mean"] * res["epochs"]:.2f}s ')

    acc = test(amodel, test_loader, device, make_adv=False)
    print(f"Clean accuracy: {acc:.4f}%")

    # PGD eval
    attack_kwargs = {
        "constraint": args.constraint,
        "eps": args.eps,
        "step_size": args.step_size,
        "iterations": args.pgd_iters,
        "random_start": (args.random_restarts > 0),
        "random_restarts": args.random_restarts,
        "do_tqdm": False
    }
    acc = test(amodel, test_loader, device, make_adv=True, attack_kwargs=attack_kwargs)
    print(f"Adversarial accuracy (eps: {args.eps}): {acc:.4f}%")


if __name__ == "__main__":
    main()

