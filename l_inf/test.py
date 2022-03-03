import os
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

    args = get_args()
    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    _, test_loader = get_loaders(args.dataroot, args.batch_size)

    # Training time statistics
    logfile = f'{args.load_path.replace(args.load_path.split("/")[-1], "output.log")}'
    if not os.path.exists(logfile):
        print("Unable to find log file, skipping train time stats computation !!!")
    else:
        res = compute_time_stats(logfile)
        print(f'Average epoch time: {res["mean"]:.2f}s, 95% confidence interval: {res["ci"]:.2f}s')
        print(f'Total training time: {res["mean"] * res["epochs"]/3600:.2f}h or {res["mean"] * res["epochs"]/60:.2f}m or {res["mean"] * res["epochs"]:.2f}s ')

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
    print(f'Clean Loss: {test_loss:.4f} \t Clean Acc: {test_acc:.4f}')

    # epsilon and attack step_size hard-coded in function
    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, args.pgd_iters, args.random_restarts, device)
    print(f'Adversarial Loss: {pgd_loss:.4f} \t Adversarial Acc: {pgd_acc:.4f}')


if __name__ == "__main__":
    main()
