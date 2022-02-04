import numpy as np
import scipy.stats


## RRM resnet18 => resnet50
#read_file = [
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=24',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=24_rerun1',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=24_rerun2',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=48',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=48_rerun1',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=48_rerun2',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=72',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=72_rerun1',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=72_rerun2',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=96',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=96_rerun1',
#    'cifar10_resnet18_to_resnet50_rrm_with_replay=1_epochs=96_rerun2',
#]


## RRM vgg11 => resnet50
#read_file = ['cifar10_vgg11_to_resnet50_rrm_with_replay=1_epochs=48_lambda=0.005']


# Free AT resnet50 (epochs = 48)
#read_file = [
#    'cifar10_resnet50_free_at_m=8_e=3',
#    'cifar10_resnet50_free_at_m=8_e=3_rerun1',
#    'cifar10_resnet50_free_at_m=8_e=3_rerun2',
#    'cifar10_resnet50_free_at_m=8_e=6',
#    'cifar10_resnet50_free_at_m=8_e=6_rerun1',
#    'cifar10_resnet50_free_at_m=8_e=6_rerun2'
#]

## Free AT resnet50 (epochs = 96)
#read_file = [
#    'cifar10_resnet50_free_at_m=8_e=9',
#    'cifar10_resnet50_free_at_m=8_e=9_rerun1',
#    'cifar10_resnet50_free_at_m=8_e=9_rerun2',
#    'cifar10_resnet50_free_at_m=8_e=12',
#    'cifar10_resnet50_free_at_m=8_e=12_rerun1',
#    'cifar10_resnet50_free_at_m=8_e=12_rerun2'
#]


# Fast AT resnet50
read_file = [ 'cifar10_resnet50_pgd_at']



all_vals = []
for r in read_file:
    print(r)
    data = open(f'./checkpoints/{r}/output.log', 'r').read().split('\n')[:-1]

    for i, d in enumerate(data):
#        if 'Epoch' in d:
#            tmp = d.split('\t')
#            if 'free' in r:
#                all_vals.append(float(tmp[1].split(':')[-1].strip())/8)
#            else:
#                all_vals.append(float(tmp[1].split(':')[-1].strip()))
#

        if 'Total train time' in d:
            break

        if i > 1:
            tmp = d.split('\t')
            all_vals.append(float(tmp[1]))




all_vals = np.array(all_vals)
pop_size = len(all_vals)
pop_mean = np.mean(all_vals)
pop_std = np.std(all_vals)
#print(np.mean(all_vals), np.std(all_vals))
#
#sample_size = 49
#sample = np.random.choice(all_vals, size=sample_size, replace=False)
#sample_mean = np.mean(sample)
#print('a', np.std(sample))
##print(pop_mean, sample_mean)
#
#z_score = (sample_mean - pop_mean) / (pop_std / np.sqrt(sample_size))
#
## two-tailed
#p_val = scipy.stats.norm.sf(abs(z_score))*2
#print(f'If population mean is {pop_mean:.6f} then the probability of the sample mean being {sample_mean:.6f} is {p_val:.4f}.')
#print(f'Sample mean is {z_score} standard deviations away from population mean.')

std_err = pop_std / np.sqrt(pop_size)
ci = 1.962 * std_err
print(ci)
