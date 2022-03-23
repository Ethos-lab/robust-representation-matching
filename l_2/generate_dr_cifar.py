import os
import tqdm
import time
import argparse
import numpy as np

import torch

from robustness import model_utils
from robustness.datasets import DATASETS
from robustness.tools import helpers


def get_args():
    parser = argparse.ArgumentParser(description="Robust data generation script.")
    parser.add_argument("--dataset", type=str, choices=["cifar"],
                        default="cifar", help="dataset")
    parser.add_argument("--dataroot", type=str, default="./CIFAR10",
                        help="path to dataset")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints")
    parser.add_argument("--model-dir", type=str, default="cifar_resnet18_l2_1_0")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--batch-size", type=int, default=500, help="initial batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="for dataloader")
    # attack hyperparams (no default values, pass every time)
    parser.add_argument("--iterations", type=int, default=1000, help="perturb number of steps")
    parser.add_argument("--step-size", type=float, default=0.1, help="perturb step size")
    parser.add_argument("--constraint", type=str, choices=["2", "inf"], default="2",
                        help="PGD attack constraint metric")

    args = parser.parse_args()

    return args


def main():
    device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')

    # Get arguments
    args = get_args()

    #NOTE: Change here if you want to store robust data somewhere else
    args.save_dir = f"./robust_data_cifar/{args.model_dir}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for arg, value in sorted(vars(args).items()):
        print(f"Argument {arg}: {value}")

    # Load dataset
    dataset_function = DATASETS[args.dataset]
    dataset = dataset_function(args.dataroot)
    train_loader, _ = dataset.make_loaders(data_aug=False,
                                           batch_size=args.batch_size,
                                           workers=args.num_workers)
    if torch.cuda.device_count() > 0:
        train_loader = helpers.DataPrefetcher(train_loader)

    # Load model
    model_kwargs = {
        "arch": args.arch,
        "dataset": dataset,
        "resume_path": f"{args.ckpt_dir}/{args.model_dir}/checkpoint.pt.best"
    }
    model, _ = model_utils.make_and_restore_model(**model_kwargs)
    model.eval()

    # Custom loss for inversion
    def inversion_loss(model, inp, targ):
        _, rep = model(inp, with_latent=True)
        loss = torch.norm(rep - targ, dim=1)
        return loss, None

    # PGD parameters
    kwargs = {
        "custom_loss": inversion_loss,
        "constraint": args.constraint,
        "eps": 1e3, #no restriction on how much to change image
        "step_size": args.step_size,
        "iterations": args.iterations,
        "do_tqdm": False,
        "targeted": True,
        "use_best": True
    }

    # Select images to invert (random samples from the test set)
    xadvs = []
    labels = []
    start_t = time.time()
    pbar = tqdm.tqdm(total=len(train_loader))
    for i, (im, label) in enumerate(train_loader):
        # get random init points
        im_n = []
        for _ in range(im.shape[0]):
            idx = int(np.random.randint(0, len(train_loader.dataset)))
            im_n.append(train_loader.dataset.__getitem__(idx)[0])
        im_n = torch.stack(im_n)

        # get target feature vector
        with torch.no_grad():
            (_, rep), _ = model(im.to(device), with_latent=True)

        # perform pgd
        _, xadv = model(im_n.to(device), rep.clone(), make_adv=True, **kwargs)

        xadvs.append(xadv.detach().cpu())
        labels.append(label.cpu())
        pbar.update(1)
    pbar.close()

    # save data to file
    torch.save(xadvs, f"{args.save_dir}/{args.dataset.upper()}_ims")
    torch.save(labels, f"{args.save_dir}/{args.dataset.upper()}_lab")
    print(f"Total duration: {time.time() - start_t} secs")


if __name__ == "__main__":
    main()
