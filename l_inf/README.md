In this set of experiments, our threat model consists of a l_infinity bound adversary with `epsilon = 8`. Please run all the commands below from the project root directory.

## Training
The following are the commands to train the models from Table 1 in the main paper:

**1. SAT**

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
	--arch resnet50 \						# model architecutre
    --out-dir checkpoints \ 				# directory where all checkpoints will be stored
    --exp-name cifar_resnet50_linf_8_0 \ 	# name of checkpoint save dir for this experiment
    --adv-train 1 \ 						# perform adversarial training
    --adv-eval 1 \ 							# perform adversarial evaluation at the end of every epoch
    --constraint inf \ 						# l_2 adversay
	--eps 0.03137254901960784 \ 			# adversary budget (8/255)
    --attack-lr 0.00784313725490196 \ 		# attack step size (2/255)
    --random-start 1 						# start attack at a point randomly sampled from the neighborhood of the given input
```


**2. fast AT**

```
# ResNet50 fast AT train
python -m l_inf.train_pgd --dataroot /path/to/cifar --arch resnet50 --opt-level O2
```

**3. free AT**

```
# ResNet50 free AT train
python -m l_inf.train_free --dataroot /path/to/cifar --arch resnet50  --opt-level O2
```

**4. RRM (VGG11)**

- Step 1: Train the VGG11 teacher using fast AT:

```
# VGG11 fast AT train
python -m l_inf.train_pgd --dataroot /path/to/cifar --arch vgg11 --opt-level O2
```

- Step 2: Train the ResNet50 student using RRM and the previously trained VGG11 model:

```
# RRM (VGG11) train
python -m l_inf.train_rrm --dataroot /path/to/cifar --arch resnet50 --t-arch vgg11 --t-loadpath /path/to/vgg11/teacher/checkpoint.pt --feat-loss cosine --xent-weight 0.005 --feat-weight -1
```

**4. RRM (ResNet18)**

- Step 1: Train the ResNet18 teacher using fast AT:

```
# ResNet18 fast AT train
python -m l_inf.train_pgd --dataroot /path/to/cifar --arch resnet18 --opt-level O2
```

- Step 2: Train the ResNet50 student using RRM and the previously trained ResNet18 model:

```
# RRM (ResNet18) train
python -m l_inf.train_rrm --dataroot /path/to/cifar --arch resnet50 --t-arch resnet18 --t-loadpath /path/to/resnet18/teacher/checkpoint.pt --feat-loss cosine --xent-weight 0.005 --feat-weight -1
```

## Evaluation

We provide two evaluation scripts in this repo:

* `test.py`: uses a manual implementation of the PGD attack.

```
# PGD Attack
python -m l_inf.test --arch resnet50 --load-path /path/to/checkpoint.pt --dataroot /path/to/cifar --random-restarts 10 --pgd-iters 50
```

* `ibm_test.py`: uses [IBM's ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to execute sate-of-the-art attacks like AutoPGD.

```
# PGD Attack
python -m l_inf.ibm_test --arch resnet50 --load-path /path/to/checkpoint.pt --dataroot /path/to/cifar --attack pgd --random-restarts 10 --pgd-iters 50

# AutoPGD Attack (reported in main paper)
python -m l_inf.ibm_test --arch resnet50 --load-path /path/to/checkpoint.pt --dataroot /path/to/cifar --attack auto_pgd --random-restarts 10 --pgd-iters 50

```

Both these scripts return accuracy on the clean test set by default. Run the AutoPGD attack by appropriately setting the args `arch`, `load-path`, `data-path`, to get the numbers reported in Table 1.