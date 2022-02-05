In this set of experiments, our threat model consists of a l_infinity bound adversary with epsilon = 8 and attack step size = 2.
## Training
To train a resnet50 model on CIFAR-10 dataset, refer to the following example commands:

**1. fast SAT**

```python train_pgd.py --data-dir /path/to/cifar --arch resnet50```

**2. fast free-AT**

```python train_free.py --data-dir /path/to/cifar --arch resnet50```

**3. RRM (VGG11)**

- Train the VGG11 teacher using fast SAT:

```python train_pgd.py --data-dir /path/to/cifar --arch vgg11```

- Train the ResNet50 student using RRM and the previously trained VGG11 model:

```python train_rrm.py --data-dir /path/to/cifar --arch resnet50 --t-arch vgg11 --t-loadpath /path/to/teacher/checkpoint.pt.last --feat-loss cosine --xent-weight 0.005 --feat-weight -1```

## Evaluation

We provide two evaluation scripts in this repo:

* `test.py`: uses a manual implementation of the PGD attack.

```
# PGD Attack
python test.py --arch resnet50 --load-path /path/to/checkpoint.pt.last --data-dir /path/to/cifar --random-restarts 10 --pgd-iters 50
```

* `ibm_test.py`: uses [IBM's ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to execute sate-of-the-art attacks like AutoPGD.

```
# PGD Attack
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt.last --data-dir /path/to/cifar --attack pgd --random-restarts 10 --pgd-iters 50

# AutoPGD Attack
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt.last --data-dir /path/to/cifar --attack auto_pgd --random-restarts 10 --pgd-iters 50

```

Both these scripts return accuracy on the clean test set by default.