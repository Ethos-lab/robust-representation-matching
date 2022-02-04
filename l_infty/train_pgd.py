import argparse
import logging
import os
import time
import tqdm

import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (upper_limit, lower_limit, clamp, std, get_loaders, evaluate_pgd,
  evaluate_standard, cifar10_mean, cifar10_std, ModelwithInputNormalization)
import cifar_models

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Fast SAT training script for CIFAR-10.')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/media/big_hdd/data/CIFAR10', type=str)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--attack-iters', default=7, type=int, help='Attack iterations')
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--alpha', default=2, type=int, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', default='checkpoints', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument("--arch", type=str, default="resnet50")
    parser.add_argument('--exp-name', default='', type=str)

    return parser.parse_args()


def main():
    args = get_args()

    args.out_dir = f'{args.out_dir}/cifar10_{args.arch}_pgd_at'
    if args.exp_name != '':
        args.out_dir += args.exp_name

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / torch.ones_like(std)
    alpha = (args.alpha / 255.) / torch.ones_like(std)

    model = cifar_models.__dict__[args.arch](num_classes=10)
    model = ModelwithInputNormalization(model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model = model.cuda()
    model.train()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    # Training
    start_train_time = time.time()
    logger.info('Epoch \t Seconds \t LR \t \t Train Loss \t Train Acc')
    for epoch in range(args.epochs):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        pbar = tqdm.tqdm(total=len(train_loader))
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()
            delta = torch.zeros_like(X).cuda()
            if args.delta_init == 'random':
                for ii in range(len(epsilon)):
                    delta[:, ii, :, :].uniform_(-epsilon[ii][0][0].item(), epsilon[ii][0][0].item())
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
            delta.requires_grad = True
            for _ in range(args.attack_iters):
                output = model(X + delta)
                loss = criterion(output, y)
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.grad.zero_()
            delta = delta.detach()
            output = model(X + delta)
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            scheduler.step()
            pbar.set_description(f'Epoch: {epoch}/{args.epochs}, Iter: {i}/{len(train_loader)}, Loss: {train_loss/train_n:.4f}, Accuracy: {100*train_acc/train_n:.4f}%')
            pbar.update(1)
        pbar.close()
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('%d \t %.1f \t \t %.4f \t %.4f \t %.4f',
            epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)
        torch.save({
                "state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
            },
            os.path.join(args.out_dir, f"checkpoint.pt.{epoch}")
        )

    train_time = time.time()
    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = cifar_models.__dict__[args.arch](num_classes=10)
    model_test = ModelwithInputNormalization(model_test, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model_test = model_test.cuda()

    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 1)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)


if __name__ == "__main__":
    main()
