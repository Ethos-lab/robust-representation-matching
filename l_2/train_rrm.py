import os
import sys
import tqdm
import time
import logging
import argparse
import numpy as np
from apex import amp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from robustness import cifar_models, imagenet_models
from robustness.tools import helpers, folder
from robustness.datasets import DATASETS
import robustness.data_augmentation as da

from utils import load_checkpoint, ModelwithInputNormalization

torch.backends.cudnn.benchmark = True

CIFAR_COMPAT_MODELS = ["dnn", "vgg11", "vgg19", "resnet18", "resnet50"]
IMGNET_COMPAT_MODELS = ["alexnet", "vgg16", "resnet50"]


def get_args():
    parser = argparse.ArgumentParser(description="Training script for CIFAR10.")
    parser.add_argument("--dataset", type=str, choices=["cifar", "restricted_imagenet"],
                        default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="./data",
                        help="path to dataset")
    parser.add_argument("--rob-dataroot", type=str, default="./datasets/d_robust_CIFAR",
                        help="path to robust dataset")
    parser.add_argument("--load-std-data", action="store_true",
                        help="loads D if true, else loads D_r")
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
    parser.add_argument('--lr-schedule', default='cosine', type=str, choices=['multistep', 'cosine'])
    parser.add_argument("--batch-size", type=int, default=128, help="initial batch size")
    parser.add_argument("--weight-decay", type=float, default=0.00005, help="weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    parser.add_argument("--xent-weight", type=float, default=1.)
    parser.add_argument("--feat-weight", type=float, default=1.)
    parser.add_argument("--feat-loss", type=str, choices=["l2", "cosine"], default="cosine")
    parser.add_argument("--mixed-prec", action="store_true")
    # testing hyperparams
    parser.add_argument("--val-batch-size", type=int, default=128, help="initial batch size")

    args = parser.parse_args()

    return args


def train(args, s_model, t_model, trainloader, valloader, criterion, optimizer, scheduler, logger,
          start_epoch, r_time_avg):

    # average meters for tracking losses
    loss_meter = helpers.AverageMeter()
    xent_meter = helpers.AverageMeter()
    fl_meter = helpers.AverageMeter()

    if args.feat_loss == 'cosine':
        criterion_feat = nn.CosineSimilarity(dim=1, eps=1e-6)

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
        pbar = tqdm.tqdm(total=num_batches)
        for i, data in enumerate(trainloader, 0):
            images, labels = data
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            curr_bs = images.size(0)
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs, rep_std = s_model(images, with_latent=True)
            if args.feat_weight != 0:
                with torch.no_grad():
                    _, rep_rob = t_model(images, with_latent=True)
                if args.feat_loss == 'l2':
                    fl = torch.mean(torch.div(
                             torch.norm(rep_std - rep_rob.detach(), dim=1),
                             torch.norm(rep_rob.detach(), dim=1)
                         ))
                elif args.feat_loss == 'cosine':
                    fl = torch.mean(criterion_feat(rep_std, rep_rob))
                else:
                    raise ValueError
            else:
                fl = torch.tensor(0)

            xent = criterion(outputs, labels)

            loss = args.xent_weight*xent + args.feat_weight*fl
            
            if args.mixed_prec:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), curr_bs)
            xent_meter.update(xent.item(), curr_bs)
            fl_meter.update(fl.item(), curr_bs)

            _, predicted = torch.max(outputs.data, 1)
            total += curr_bs
            correct += (predicted == labels).sum().item()

            pbar.set_description(
                f"Train | loss: {loss_meter.avg:.4f} ({xent_meter.avg:.4f}, {fl_meter.avg:.4f})")
            scheduler.step()
            pbar.update(1)
        pbar.close()

        # update running time average
        curr_time = time.time() - start_t
        if epoch == 0:
            r_time_avg = curr_time
        else:
            r_time_avg = r_time_avg + ((curr_time - r_time_avg) / epoch)

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']

        # evaluate at end of each epoch
        train_acc = 100 * correct / total
        logger.info(f"Train (lr: {curr_lr} ) | "+\
            f"loss: {loss_meter.avg:.4f} ({xent_meter.avg:.4f}, {fl_meter.avg:.4f}) " +\
            f"std accuracy: {train_acc:.4f}%")

        val_acc = test(s_model, valloader)
        logger.info(f"Val | std accuracy: {val_acc:.4f}%")

    logger.info(f"Finished Training (@ {r_time_avg:.4f} secs/epoch)!!!")

    # save last checkpoint
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



def test(model, valloader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

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

    if not args.load_std_data: assert (args.feat_weight == 0.0 and args.xent_weight == 1.0),\
        "When robust dataset (D_r) is loaded, can only train with xent loss !!!"

    if args.save_dir == "":
        if args.feat_weight == 0:
            assert args.xent_weight == 1.0
            args.save_dir = f"./checkpoints/{args.dataset}_{args.student_arch}"
        else:
            args.save_dir = f"./checkpoints/{args.dataset}-feat_loss={args.feat_loss}-teacher={args.teacher_arch}-student={args.student_arch}-xent_weight={args.xent_weight}-feat_weight={args.feat_weight}"

        if args.exp_name != "":
           args.save_dir += f"-{args.exp_name}"
    else:
        if args.feat_weight == 0:
            assert args.xent_weight == 1.0

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
    if args.load_std_data:
        # Load standard dataset (D) using arg: dataroot.
        train_loader, val_loader = dataset.make_loaders(workers=args.num_workers,
                                                        batch_size=args.batch_size,
                                                        val_batch_size=args.val_batch_size,
                                                        data_aug=True,
                                                        shuffle_train=True,
                                                        shuffle_val=False)
    else:
        assert args.dataset == "cifar", "Robust data training only supported for CIFAR10"
        # Load robust dataset (D_r) using arg: rob_dataroot.
        train_data = torch.cat(torch.load(os.path.join(args.rob_dataroot, "CIFAR_ims"))).cpu()
        train_labels = torch.cat(torch.load(os.path.join(args.rob_dataroot, "CIFAR_lab"))).cpu()

        train_set = folder.TensorDataset(train_data, train_labels,
                                         transform=da.TRAIN_TRANSFORMS_DEFAULT(32))
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)

        #NOTE: Still need arg: dataroot to load val set.
        _, val_loader = dataset.make_loaders(only_val=True,
                                            batch_size=args.val_batch_size,
                                            shuffle_val=False,
                                            workers=args.num_workers)

    train_loader = helpers.DataPrefetcher(train_loader)
    val_loader = helpers.DataPrefetcher(val_loader)

    # Prepare teacher model
    if args.feat_weight != 0:
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
    else:
        t_model = None

    # Prepare student model
    if args.dataset == "cifar":
        assert args.student_arch in CIFAR_COMPAT_MODELS, f"Please use one of these models for student: {CIFAR_COMPAT_MODELS}"
        s_model = cifar_models.__dict__[args.student_arch](num_classes=10)
    else:
        assert args.student_arch in IMGNET_COMPAT_MODELS, f"Please use one of these models for student: {IMGNET_COMPAT_MODELS}"
        s_model = imagenet_models.__dict__[args.student_arch](num_classes=9)

    s_model = ModelwithInputNormalization(s_model, dataset.mean, dataset.std).cuda()

    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

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

    if args.mixed_prec:
        opt_level = 'O2' if args.dataset == "cifar" else 'O1'
        print(opt_level)
        [s_model, t_model], optimizer = amp.initialize([s_model, t_model], optimizer, opt_level=opt_level)
 
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
    train(args, s_model, t_model, train_loader, val_loader, criterion, optimizer, scheduler,
          logger, start_epoch, r_time_avg)

if __name__ == "__main__":
    main()
