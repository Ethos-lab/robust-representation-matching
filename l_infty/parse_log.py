import numpy as np

read_file = './checkpoints/cifar10_vgg11_to_resnet50_rrm_with_replay=1_epochs=48_lambda=0.005/output.log'

data = open(read_file, 'r').read().split('\n')[:-1]

###################################################

#avg = 0
#ctr = 0
#for i,d in enumerate(data):
#    if 'Total train time' in d:
#        break
#
#    if i > 1:
#        tmp = d.split('\t')
#        avg += float(tmp[1])
#        ctr += 1

###################################################

all_vals = []
for d in data:
    if 'Epoch' in d:
        tmp = d.split('\t')
        all_vals.append(float(tmp[1].split(':')[-1].strip()))

avg = np.mean(all_vals)
###################################################

if 'free_at' in read_file:
    avg /= 8

print(avg)
