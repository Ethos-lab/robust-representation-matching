import os
import sys
import tqdm
import time
import logging
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from robustness import cifar_models, imagenet_models
from robustness.tools import helpers
from robustness.datasets import DATASETS

from utils import load_checkpoint, ModelwithInputNormalization


CIFAR_COMPAT_MODELS = ["vgg11", "vgg19", "resnet18", "resnet50", "mobilenetv2"]
IMGNET_COMPAT_MODELS = ["alexnet", "vgg16", "resnet50"]

def get_args():
    parser = argparse.ArgumentParser(description="Student-teacher training script for CIFAR10 using KD loss.")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"],
                        default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="./data",
                        help="path to dataset")
    parser.add_argument("--student-arch", type=str, default="resnet18")
    parser.add_argument("--teacher-load-path", type=str,
                        default="/path/to/teacher/checkpoint.pt.best")
    parser.add_argument("--teacher-arch", type=str, default="resnet50")
    parser.add_argument("--exp-name", type=str, default="",
                        help="additional description for exp to add to save_dir name")
    parser.add_argument("--save-dir", type=str, default="",
                        help="overwrites automatically created save_dir")
    parser.add_argument("--resume", type=str, default="",
                        help="resume training from the checkpoint saved at this path")
    # training hyperparams
    parser.add_argument("--lr", type=float, default=0.1, help="initial learning rate")
    parser.add_argument("--batch-size", type=int, default=128, help="initial batch size")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="gamma of optim.lr_scheduler.StepLR, decay of lr")
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--temperature", type=float, default=30.)
    # testing hyperparams
    parser.add_argument("--val-batch-size", type=int, default=128, help="initial batch size")

    args = parser.parse_args()

    return args


def train(args, s_model, t_model, trainloader, valloader, optimizer, scheduler, logger, start_epoch, r_time_avg):
    # average meters for tracking losses
    loss_meter = helpers.AverageMeter()
    xent_meter = helpers.AverageMeter()
    fl_meter = helpers.AverageMeter()

    KL_loss = nn.KLDivLoss()
    XENT_loss = nn.CrossEntropyLoss()

    num_batches = len(trainloader)
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch: {epoch}/{args.epochs} (@ {r_time_avg:.4f} secs/epoch)")
        s_model.train()

        # Init counters
        total = 0.0
        correct = 0.0
        loss_meter.reset()
        xent_meter.reset()
        fl_meter.reset()

        start_t = time.time()
        pbar = tqdm.tqdm(total=num_batches, leave=False)
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            curr_bs = images.size(0)

            # zero the parameter gradients
            optimizer.zero_grad()

            with torch.no_grad():
                t_outputs = t_model(images)
            s_outputs = s_model(images)

            fl = KL_loss(F.log_softmax(s_outputs/args.temperature, dim=1), F.softmax(t_outputs/args.temperature, dim=1))
            xent = XENT_loss(s_outputs/args.temperature, labels)
            loss = args.alpha*args.temperature*args.temperature*fl + (1.0-args.alpha)*xent

            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss.item(), curr_bs)
            xent_meter.update(xent.item(), curr_bs)
            fl_meter.update(fl.item(), curr_bs)

            _, predicted = torch.max(s_outputs.data, 1)
            total += curr_bs
            correct += (predicted == labels).sum().item()

            pbar.set_description(
                f"Train | loss: {loss_meter.avg:.4f} ({xent_meter.avg:.4f}, {fl_meter.avg:.4f})")

            pbar.update(1)
        pbar.close()

        # update running time average
        curr_time = time.time() - start_t
        if epoch == 0:
            r_time_avg = curr_time
        else:
            r_time_avg = r_time_avg + ((curr_time - r_time_avg) / epoch)

        # evaluate at end of each epoch
        train_acc = 100 * correct / total
        logger.info(f"Train | "+\
            f"loss: {loss_meter.avg:.4f} ({xent_meter.avg:.4f}, {fl_meter.avg:.4f}) " +\
            f"std accuracy: {train_acc:.4f}%")

        val_acc = test(s_model, valloader)
        logger.info(f"Val | std accuracy: {val_acc:.4f}%")

    try:
        sd = s_model.net.state_dict()
    except:
        sd = s_model.module.net.state_dict()
    torch.save({
            "state_dict": sd,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": val_acc,
            "time_per_epoch": r_time_avg
        },
        os.path.join(args.save_dir, "checkpoint.pt.last")
    )

    logger.info(f"Finished Training (@ {r_time_avg:.4f} secs/epoch)!!!")


def test(model, valloader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()

            # forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    nat_acc = 100 * correct / total

    return nat_acc


def main():
    # Setting up logger
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    args = get_args()

    if args.save_dir == "":
        args.save_dir = f"./checkpoints/{args.dataset}-kd_loss-teacher={args.teacher_arch}-student={args.student_arch}-alpha={args.alpha}-temperature={args.temperature}"
        if args.exp_name != "":
           args.save_dir += f"-{args.exp_name}"

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    print(f'Saving checkpoint at: {args.save_dir} !!!')

    logfile_path = os.path.join(args.save_dir, "log.txt")
    logger.addHandler(logging.FileHandler(logfile_path))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    dataset_function = DATASETS[args.dataset]
    dataset = dataset_function(args.dataroot)
    train_loader, val_loader = dataset.make_loaders(workers=args.num_workers,
                                                    batch_size=args.batch_size,
                                                    val_batch_size=args.val_batch_size,
                                                    data_aug=True,
                                                    shuffle_train=True,
                                                    shuffle_val=False)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)

    # Prepare teacher model
    if args.dataset == "cifar":
        assert args.teacher_arch in CIFAR_COMPAT_MODELS, f"Please use one of these models for teacher: {CIFAR_COMPAT_MODELS}"
        t_model = cifar_models.__dict__[args.teacher_arch](num_classes=10)
    else:
        assert args.teacher_arch in IMGNET_COMPAT_MODELS, f"Please use one of these models for teacher: {IMGNET_COMPAT_MODELS}"
        t_model = imagenet_models.__dict__[args.teacher_arch](num_classes=9)

    ckpt = load_checkpoint(args.teacher_load_path)
    t_model.load_state_dict(ckpt["model_sd"])
    print(f"Successfully loaded teacher checkpoint from epoch: {ckpt['load_epoch']} !!!")
    del ckpt

    t_model = ModelwithInputNormalization(t_model, dataset.mean, dataset.std).cuda()
    t_model.eval()

    # Prepare student model
    if args.dataset == "cifar":
        assert args.student_arch in CIFAR_COMPAT_MODELS, f"Please use one of these models for student: {CIFAR_COMPAT_MODELS}"
        s_model = cifar_models.__dict__[args.student_arch](num_classes=10)
    else:
        assert args.student_arch in IMGNET_COMPAT_MODELS, f"Please use one of these models for student: {IMGNET_COMPAT_MODELS}"
        s_model = imagenet_models.__dict__[args.student_arch](num_classes=9)

    s_model = ModelwithInputNormalization(s_model, dataset.mean, dataset.std).cuda()

    # optimizer
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    # scheduler
    lr_steps = args.epochs * len(train_loader)
    if args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    elif args.lr_schedule == 'cosine':
        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos(step / total_steps * np.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                lr_steps,
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.lr))
    else:
        scheduler = None

    # Run on multiple GPUs
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        t_model = nn.DataParallel(t_model)
        s_model = nn.DataParallel(s_model)

    # Resume student training
    if args.resume != "":
        assert os.path.exists(args.resume), f"Invalid load path: {args.resume}"
        ckpt = load_checkpoint(args.resume, mode="train")
        #TODO: better way to do this?
        try:
            s_model.net.load_state_dict(ckpt["model_sd"])
        except:
            s_model.module.net.load_state_dict(ckpt["model_sd"])
 
        s_model.net.load_state_dict(ckpt["model_sd"])
        optimizer.load_state_dict(ckpt["optim_sd"])
        start_epoch = ckpt["load_epoch"] + 1
        r_time_avg = ckpt["r_time_avg"]
        del ckpt

        logger.info(f"Resuming training from epoch {start_epoch} !!!")
    else:
        start_epoch = 0
        r_time_avg = 0

    logger.info("Started Training !!!")
    train(args, s_model, t_model, train_loader, val_loader, optimizer, scheduler, logger, start_epoch, r_time_avg)

if __name__ == "__main__":
    main()


