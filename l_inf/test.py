import sys
sys.path.append('../')

import torch
import logging
import argparse
import numpy as np

try:
    from robustness import cifar_models
    from l_inf.utils import (get_loaders, evaluate_pgd, evaluate_standard, cifar10_mean, cifar10_std,
                             ModelwithInputNormalization, load_checkpoint)
except:
    raise ValueError("Make sure to run with python -m from root project directory")

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Testing script using manual attack implementations.")
    parser.add_argument("--dataroot", type=str, default="./CIFAR10",
                        help="path to dataset if you need diff from default")
    parser.add_argument("--arch", type=str, default="resnet50", choices=['resnet18', 'resnet50', 'vgg11', 'vgg19'])
    parser.add_argument("--load-path", type=str, default="checkpoint.pt.best")
    parser.add_argument("--batch-size", type=int, default=100, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--pgd-iters", type=int, default=50, help="for PGD")
    parser.add_argument("--random-restarts", type=int, default=10, help="for PGD")
 
    return parser.parse_args()


def main():
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    args = get_args()
    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    _, test_loader = get_loaders(args.dataroot, args.batch_size)

    # Evaluation
    model_test = cifar_models.__dict__[args.arch](num_classes=10)
    model_test = ModelwithInputNormalization(model_test, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model_test = model_test.to(device)

    # model_test.load_state_dict(torch.load(args.load_path)['state_dict'])
    ckpt = load_checkpoint(args.load_path)
    model_test.net.load_state_dict(ckpt['model_sd'])
    model_test.float()
    model_test.eval()

    test_loss, test_acc = evaluate_standard(test_loader, model_test, device)
    print(f'Test Loss: {test_loss:.4f} \t Test Acc: {test_acc:.4f}')

    # epsilon and attack step_size hard-coded in function
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, args.pgd_iters, args.random_restarts, device)
    print(f'PGD Loss: {pgd_loss:.4f} \t PGD Acc: {pgd_acc:.4f}')


if __name__ == "__main__":
    main()
