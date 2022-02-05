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
from utils import (get_loaders, evaluate_pgd, evaluate_standard, cifar10_mean, cifar10_std, ModelwithInputNormalization)


logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description='RRM training script for CIFAR-10.')
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='/media/big_hdd/data/CIFAR10', type=str)
    parser.add_argument('--epochs', default=48, type=int, help='Total number of epochs will be this argument * number of minibatch replays.')
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep', 'cosine'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--minibatch-replays', default=1, type=int)
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
    # method-specific args
    parser.add_argument("--t-arch", type=str, default="vgg11", choices=['resnet18', 'resnet50', 'vgg11', 'vgg19'])
    parser.add_argument("--t-load-path", type=str, default="")
    parser.add_argument("--xent-weight", type=float, default=1.)
    parser.add_argument("--feat-loss", type=str, default="cosine", choices=["l2", "cosine"])
    parser.add_argument("--feat-weight", type=float, default=1.)

    return parser.parse_args()


def main():
    args = get_args()

    args.out_dir = f'{args.out_dir}/cifar10_{args.t_arch}_to_{args.arch}_rrm_with_replay={args.minibatch_replays}_epochs={args.epochs}_lambda={args.xent_weight}'
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

    model = cifar_models.__dict__[args.arch](num_classes=10)
    model = ModelwithInputNormalization(model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model = model.cuda()

    t_model = cifar_models.__dict__[args.t_arch](num_classes=10)
    t_model.linear = None # remove final layer
    t_model = ModelwithInputNormalization(t_model, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    t_model = t_model.cuda()
    # Loading model checkpoint
    ckpt = torch.load(args.t_load_path) #load_checkpoint(args.t_load_path)
    t_model.load_state_dict(ckpt['state_dict'])
    print(f"Successfully loaded teacher from epoch: {ckpt['epoch']} !!!")
    del ckpt
    t_model.eval()

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    [model, t_model], opt = amp.initialize([model, t_model], opt, **amp_args)
    criterion_cls = nn.CrossEntropyLoss()
    if args.feat_loss == 'cosine':
        criterion_feat = nn.CosineSimilarity(dim=1, eps=1e-6)

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
        pbar = tqdm.tqdm(total=len(train_loader), leave=False)
        for i, (X, y) in enumerate(train_loader):
            X, y = X.cuda(), y.cuda()

            # forward pass through teacher
            with torch.no_grad():
                _, rep_rob = t_model(X, with_latent=True)
                rep_rob = rep_rob.detach()

            # Similar training loop style as free AT. For standard training loop, set minibatch_replays = 1.
            for _ in range(args.minibatch_replays):
                # forward pass through student
                output, rep_std = model(X, with_latent=True)

                # loss computation
                xent_loss = criterion_cls(output, y)
                if args.feat_loss == 'cosine':
                    feat_loss = torch.mean(criterion_feat(rep_std, rep_rob.detach()))
                else:
                    feat_loss = torch.mean(torch.norm(rep_std - rep_rob.detach(), dim=1))
                loss = args.xent_weight*xent_loss + args.feat_weight*feat_loss

                # backprop + weight update
                opt.zero_grad()
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                opt.step()
                scheduler.step()

            # tracking train stats
            train_loss += loss.item() * y.size(0)
            train_acc += (output.max(1)[1] == y).sum().item()
            train_n += y.size(0)
            pbar.set_description(f'Epoch: {epoch}/{args.epochs}, Iter: {i}/{len(train_loader)}, Loss: {train_loss/train_n:.4f}, Accuracy: {100*train_acc/train_n:.4f}%')
            pbar.update(1)

        pbar.close()
        epoch_time = time.time()
        lr = scheduler.get_last_lr()[0]
        # display end of epoch stats
        logger.info('Train | Epoch: %d \t Time: %.1f \t LR: %.4f \t Loss: %.4f \t Accuracy: %.4f', epoch, epoch_time - start_epoch_time, lr, train_loss/train_n, train_acc/train_n)

        # run epoch end evaluation
        model.eval()
        test_loss, test_acc = evaluate_standard(test_loader, model)
        pgd_acc = 0.0
        # Uncomment line below if want to run pgd eval at the end of every epoch
        # pgd_loss, pgd_acc = evaluate_pgd(test_loader, model, 10, 1)
        logger.info(f'Test | Std Acc: {test_acc:.4f}, PGD Acc: {pgd_acc:.4f}')

    train_time = time.time()
    logger.info('Total train time: %.4f minutes', (train_time - start_train_time)/60)

    # Save last epoch model checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "epoch": epoch,
    },
        os.path.join(args.out_dir, f"checkpoint.pt.last")
    )

    # Run end of training evaluation
    model_test = cifar_models.__dict__[args.arch](num_classes=10)
    model_test = ModelwithInputNormalization(model_test, torch.tensor(cifar10_mean), torch.tensor(cifar10_std))
    model_test = model_test.cuda()

    model_test.load_state_dict(model.state_dict())
    model_test.float()
    model_test.eval()

    pgd_loss, pgd_acc = evaluate_pgd(test_loader, model_test, 50, 10)
    test_loss, test_acc = evaluate_standard(test_loader, model_test)

    logger.info('Test Loss \t Test Acc \t PGD Loss \t PGD Acc')
    logger.info('%.4f \t \t %.4f \t %.4f \t %.4f', test_loss, test_acc, pgd_loss, pgd_acc)



if __name__ == "__main__":
    main()
