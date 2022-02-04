import torch
import logging
import argparse
import numpy as np

import cifar_models
from utils import (get_loaders, evaluate_pgd, evaluate_standard, cifar10_mean, cifar10_std, ModelwithInputNormalization)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Testing script using manual attack implementations.")
    parser.add_argument("--data-dir", type=str, default="/media/big_hdd/data/CIFAR10",
                        help="path to dataset if you need diff from default")
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument("--load-path", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=100, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument("--pgd-iters", type=int, default=50, help="for PGD")
    parser.add_argument("--random-restarts", type=int, default=10, help="for PGD")
 
    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    _, test_loader = get_loaders(args.data_dir, args.batch_size)

    # Evaluation
    model_test = cifar_models.__dict__[args.arch](num_classes=10)
    model_test = ModelwithInputNormalization(model_test, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model_test = model_test.cuda()

    model_test.load_state_dict(torch.load(args.load_path)['state_dict'])
    model_test.float()
    model_test.eval()

    test_loss, test_acc = evaluate_standard(test_loader, model_test)
    print(f'Test Loss: {test_loss:.4f} \t Test Acc: {test_acc:.4f}')

    # epsilon and attack step_size hard-coded in function
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, args.pgd_iters, args.random_restarts)
    print(f'PGD Loss: {pgd_loss:.4f} \t PGD Acc: {pgd_acc:.4f}')


if __name__ == "__main__":
    main()
