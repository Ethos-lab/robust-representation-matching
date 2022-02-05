import os
import sys
import time
import tqdm
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn

import apex.amp as amp

import cifar_models
from utils import (upper_limit, lower_limit, clamp, std, get_loaders, evaluate_pgd,
  evaluate_standard, cifar10_mean, cifar10_std, ModelwithInputNormalization)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='Fast Free-AT training script for CIFAR-10.')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/media/big_hdd/data/CIFAR10', type=str)
    parser.add_argument('--epochs', default=12, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.04, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--minibatch-replays', default=8, type=int)
    parser.add_argument('--out-dir', default='checkpoints', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument("--arch", type=str, default="resnet50", choices=['resnet18', 'resnet50', 'vgg11', 'vgg19'])
    parser.add_argument('--exp-name', default='', type=str)
 
    return parser.parse_args()


def main():
    args = get_args()

    args.out_dir = f'{args.out_dir}/cifar10_{args.arch}_free_at_m={args.minibatch_replays}_e={args.epochs}'
    if args.exp_name != '':
        args.out_dir += f'_{args.exp_name}'

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    logfile = os.path.join(args.out_dir, 'output.log')
    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO,
        filename=logfile)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

    epsilon = (args.epsilon / 255.) / torch.ones_like(std)

    model = cifar_models.__dict__[args.arch](num_classes=10)
    model = ModelwithInputNormalization(model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model = model.cuda()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    delta.requires_grad = True

    lr_steps = args.epochs * len(train_loader) * args.minibatch_replays
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos(step / total_steps * np.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lr_lambda=lambda step: cosine_annealing(
                step,
                lr_steps,
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr_max))


    # Training
    start_train_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
#        t0 = 0
#        t1 = 0
#        t2 = 0
#        t3 = 0
#        t4 = 0
#        t5 = 0
#        t6 = 0
#        t7 = 0
#        with torch.autograd.profiler.profile() as prof:
        pbar = tqdm.tqdm(total=len(train_loader))
        for i, (X, y) in enumerate(train_loader):
#            st = time.time()
            X, y = X.cuda(), y.cuda()
#            t0 += (time.time() - st)
            for _ in range(args.minibatch_replays):
#                st = time.time()
                output = model(X + delta[:X.size(0)])
#                t1 += (time.time() - st)

#                st = time.time()
                loss = criterion(output, y)
#                t2 += (time.time() - st)

#                st = time.time()
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
#                t3 += (time.time() - st)

#                st = time.time()
                grad = delta.grad.detach()
                delta.data = clamp(delta + epsilon * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
#                t4 += (time.time() - st)

#                st = time.time()
                opt.step()
#                t5 += (time.time() - st)

                delta.grad.zero_()

#                st = time.time()
                scheduler.step()
#                t6 += (time.time() - st)

#            st = time.time()
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            pbar.set_description(f'Epoch: {epoch}/{args.epochs}, Iter: {i}/{len(train_loader)}, Loss: {train_loss/train_n:.4f}, Accuracy: {100*train_acc/train_n:.4f}%')
            pbar.update(1)
#            t7 += (time.time() - st)

        pbar.close()
        epoch_time = time.time()
        lr = scheduler.get_lr()[0]
        logger.info('Train | Epoch: %d \t Time: %.1f \t LR: %.4f \t Loss: %.4f \t Accuracy: %.4f', epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)

#        print(f'move x and y to gpu: {t0/args.minibatch_replays:.4f}s')
#        print(f'student forward pass: {t1/args.minibatch_replays:.4f}s')
#        print(f'loss computation: {t2/args.minibatch_replays:.4f}s')
#        print(f'backprop: {t3/args.minibatch_replays:.4f}s')
#        print(f'attack computation: {t4/args.minibatch_replays:.4f}s')
#        print(f'parameter update: {t5/args.minibatch_replays:.4f}s')
#        print(f'scheduler step: {t6/args.minibatch_replays:.4f}s')
#        print(f'counter updates: {t7/args.minibatch_replays:.4f}s')
#        print(f'total: {t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7:.4f}s')

#        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=5))

        model.eval()
        pgd_acc = 0.0
#        pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
        test_loss, test_acc = evaluate_standard(test_loader, model)
        logger.info(f'Test | Std Acc: {test_acc:.4f}, PGD Acc: {pgd_acc:.4f}')


    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Evaluation
    model_test = cifar_models.__dict__[args.arch](num_classes=10)
    model_test = ModelwithInputNormalization(model_test, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model_test = model_test.cuda()

    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 20, 1) #50, 10
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)

    torch.save({
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
        },
        os.path.join(args.out_dir, f"checkpoint.pt.last")
    )


if __name__ == "__main__":
    main()
