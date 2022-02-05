In this set of experiments, our threat model consists of a l_2 bound adversary with `epsilon = 1.0` for CIFAR-10 and `epsilon = 3.0` for Restricted-ImageNet.

## Training

To train a resnet50 model on CIFAR-10 dataset, refer to the following example commands:

**Standard Adversarial Training ([Madry et. al.](https://arxiv.org/abs/1706.06083))** against an l_2 bound PGD (eps = 1.0):

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --out-dir checkpoints \
    --exp-name cifar_resnet50_l2_1_0 \
    --arch resnet50 \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

**Robust Dataset Training ([Ilyas et. al.](https://arxiv.org/abs/1905.02175))**

This is a 3-step process:

1. Training a robust teacher (e.g., ResNet18)

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --out-dir checkpoints \
    --exp-name cifar_resnet18_l2_1_0 \
    --arch resnet18 \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

2.  Generating robust training data using teacher

	To generate robustified version of CIFAR10 training set using the adversarially trained ResNet18 model from previous step, run the command below. By default, the robust data is stored at: `./robust_data_cifar/cifar_resnet18_l2_1_0`.

```bash
python generate_dr_cifar.py \
    --dataset cifar \ # this script only supports cifar
    --dataroot /path/to/cifar \
    --batch-size 500 \
    --ckpt-dir ./checkpoints \
    --arch resnet18 \
    --model-dir cifar_resnet18_l2_1_0 \
    --step-size 0.1 \
    --iterations 1000
```


3. Training a student model on the robust training data

   To train a resnet50 model on the robustified data from previous step, run the following command:

```bash
python train.py \
    --dataset cifar \ # this mode of training only supports cifar
    --dataroot /path/to/cifar \
    --rob-dataroot ./robust_data_cifar/cifar_resnet18_l2_1_0 \
    --student-arch resnet50 \
    --xent-weight 1.0 \
    --l2-weight 0.0 \
    --exp-name dr=resnet18_l2_1_0
```


**Robust Representation Matching (RRM)**

This is a 2-step process:

1. Training a robust teacher (e.g., ResNet18)

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --out-dir checkpoints \
    --exp-name cifar_resnet18_l2_1_0 \
    --arch resnet18 \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

2. To transfer robustness from the robust ResNet18 model to a ResNet50 model, run the following command:

```bash
python train.py \
    --dataset cifar \
    --dataroot /path/to/cifar \
    --load-std-data \
    --teacher-load-path ./checkpoints/cifar_resnet18_l2_1_0/checkpoint.pt.best \
    --teacher-arch resnet18 \
    --student-arch resnet50 \
    --xent-weight 0.001 \
    --l2-weight 1.0 
```


## Evaluation

We provide two evaluation scripts in this repo:

* `test.py`: uses a manual implementation of the PGD attack.

```
# PGD Attack
python test.py --arch resnet50 --load-path /path/to/checkpoint.pt.last --data-dir /path/to/cifar --eps 1.0 --step-size 0.125 --pgd-iters 20 --constraint 2

```

* `ibm_test.py`: uses [IBM's ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to execute sate-of-the-art attacks like AutoPGD.

```
# PGD Attack
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt.last --data-dir /path/to/cifar --attack pgd --eps 1.0 --step-size 0.125 --pgd-iters 20 --constraint 2

# AutoPGD Attack
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt.last --data-dir /path/to/cifar --attack auto_pgd --eps 1.0 --pgd-iters 20 --constraint 2

```

Both these scripts return accuracy on the clean test set by default.