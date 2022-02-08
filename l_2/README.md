In this set of experiments, our threat model consists of a l_2 bound adversary with `epsilon = 1.0` for CIFAR-10 and `epsilon = 3.0` for Restricted-ImageNet.

## Replicating Results from Table 2
### Training

The following are the commands to train the models from Table 2 in the main paper:

1. **SAT**

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
	--arch resnet50 \					# model architecutre
    --out-dir checkpoints \ 			# directory where all checkpoints will be stored
    --exp-name cifar_vgg11_l2_1_0 \ 	# name of checkpoint save dir for this experiment
    --adv-train 1 \ 					# perform adversarial training
    --adv-eval 1 \ 						# perform adversarial evaluation at the end of every epoch
    --constraint 2 \ 					# l_2 adversay
	--eps 1.0 \ 						# adversary budget
    --attack-lr 0.2 \ 					# attack step size
    --random-start 1 					# start attack at a point randomly sampled from the neighborhood of the given input
```

2. **RDT**

* Step 1: Training a robust teacher (VGG11/ResNet18):


```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --arch vgg11 \	# or, resnet18
    --out-dir checkpoints \
    --exp-name cifar_vgg11_l2_1_0 \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

* Step 2:  Generating robust training data using teacher trained in previous step:

```bash
python generate_dr_cifar.py \
    --dataset cifar \ 					# this script only supports cifar
    --dataroot /path/to/cifar \
    --arch vgg11 \						# or, resnet18
    --ckpt-dir ./checkpoints \			# directory where all checkpoints will be stored
    --model-dir cifar_vgg11_l2_1_0 \	# directory within <ckpt-dir> with teacher's checkpoint
    --batch-size 500 \
    --step-size 0.1 \
    --iterations 1000

# By default, the robust data is stored at: ./robust_data_cifar/cifar_vgg11_l2_1_0
```


* Step 3: Training a student model on the robust training data generated in previous step:

```bash
python train.py \
    --dataset cifar \
    --dataroot /path/to/cifar \
    --rob-dataroot ./robust_data_cifar/cifar_vgg11_l2_1_0 \
    --student-arch resnet50 \
    --xent-weight 1.0 \
    --l2-weight 0.0 \
    --exp-name dr=vgg11_l2_1_0
```

3. **KD (VGG11/ResNet18)**

* Step 1: Training a robust teacher (VGG11/ResNet18):

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --arch vgg11 \	# or, resnet18
    --out-dir checkpoints \
    --exp-name cifar_vgg11_l2_1_0 \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

* Step 2: Transferring robustness from the teacher trained in previous step to a ResNet50 student:

```bash
python train_kdloss.py \
    --dataset cifar \
    --dataroot /path/to/cifar \
    --load-std-data \
    --teacher-load-path ./checkpoints/cifar_vgg11_l2_1_0/checkpoint.pt \
    --teacher-arch vgg11 \
    --student-arch resnet50 \
    --alpha 1.0 \
    --temperature 30.0 
```



4. **RRM (VGG11/ResNet18)**

* Step 1: Training a robust teacher (VGG11/ResNet18):

```bash
python -m robustness.main \
    --dataset cifar \
    --data /path/to/cifar \
    --arch vgg11 \	# or, resnet18
    --out-dir checkpoints \
    --exp-name cifar_vgg11_l2_1_0 \
    --adv-train 1 \
    --adv-eval 1 \
    --constraint 2 \
    --eps 1.0 \
    --attack-lr 0.2 \
    --random-start 1
```

* Step 2: Transferring robustness from the teacher trained in previous step to a ResNet50 student:

```bash
python train.py \
    --dataset cifar \
    --dataroot /path/to/cifar \
    --load-std-data \
    --teacher-load-path ./checkpoints/cifar_vgg11_l2_1_0/checkpoint.pt \
    --teacher-arch vgg11 \
    --student-arch resnet50 \
    --xent-weight 0.001 \
    --l2-weight 1.0 
```


### Evaluation

We provide two evaluation scripts in this repo:

* `test.py`: uses a manual implementation of the PGD attack.

```
# PGD Attack
python test.py --arch resnet50 --load-path /path/to/checkpoint.pt --data-dir /path/to/cifar --eps 1.0 --step-size 0.125 --pgd-iters 20 --constraint 2

```

* `ibm_test.py`: uses [IBM's ART](https://github.com/Trusted-AI/adversarial-robustness-toolbox) to execute sate-of-the-art attacks like AutoPGD.

```
# PGD Attack
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt --data-dir /path/to/cifar --attack pgd --eps 1.0 --step-size 0.125 --pgd-iters 20 --constraint 2

# AutoPGD Attack (reported in main paper)
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt --data-dir /path/to/cifar --attack auto_pgd --eps 1.0 --pgd-iters 20 --constraint 2

```

Both these scripts return accuracy on the clean test set by default. Run the AutoPGD attack by appropriately setting args `arch`, `load-path`, `data-path`, to get the numbers reported in Table 2.


## Replicating Results from Table 3
### Training

The following are the commands to train the models from Table 3 in the main paper:

1. **SAT (ResNet50/VGG16)**

```bash
python -m robustness.main \
	--dataset restricted_imagenet \
	--data /path/to/orignal/imagenet/root \
	--arch resnet50 \	# or, vgg16
	--out-dir checkpoints \
	--exp-name rimagenet_resnet50_l2_3_0 \
	--adv-train 1 \
	--adv-eval 1 \
	--lr 0.01 \
	--step-lr 125 \
	--batch-size 128 \
	--constraint 2 \
	--eps 3.0 \
	--attack-lr 0.6 \
	--random-start 1
```

2. **RRM (AlexNet)**

This is a 2-step process:

* Step 1: Training a robust AlexNet teacher:

```bash
python -m robustness.main \
	--dataset restricted_imagenet \
	--data /path/to/orignal/imagenet/root \
	--arch alexnet \
	--out-dir checkpoints \
	--exp-name rimagenet_alexnet_l2_3_0 \
	--adv-train 1 \
	--adv-eval 1 \
	--lr 0.01 \
	--step-lr 125 \
	--batch-size 128 \
	--constraint 2 \
	--eps 3.0 \
	--attack-lr 0.6 \
	--random-start 1
```

* Step 2: Transferring robustness from the teacher trained in previous step to a ResNet50/VGG16 student:

```bash
python train.py \
    --dataset restricted_imagenet \
    --dataroot /path/to/imagenet/root \
    --load-std-data \
    --teacher-load-path /path/to/teacher/checkpoint.pt \
    --teacher-arch alexnet \
    --student-arch resnet50 \	#or, vgg16
    --xent-weight 0.001 \
    --l2-weight 1.0  \
    --epochs 60 \
    --lr-schedule 35,50
```

### Evaluation

To replicate numbers reported in Table 2, appropriately set the args `arch`, `load-path`, `data-path` in the command below:

```
# AutoPGD Attack
python ibm_test.py --arch resnet50 --load-path /path/to/checkpoint.pt --data-dir /path/to/imagenet/root --attack auto_pgd --eps 3.0 --pgd-iters 20 --constraint 2 --random-starts 5

```


