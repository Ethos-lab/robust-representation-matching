import dill

import torch
import torch.nn as nn

from robustness.tools import helpers


def load_checkpoint(load_path, mode, device):
    assert mode in ['train', 'eval']

    checkpoint = torch.load(load_path, pickle_module=dill, map_location=device)
    state_dict_path = "model"
    if not ("model" in checkpoint):
        state_dict_path = "state_dict"

    sd = checkpoint[state_dict_path]
    sd = {k.replace("module.model.", ""):v for k,v in sd.items()\
          if ("attacker" not in k) and ("normalizer" not in k)}

    load_epoch = checkpoint['epoch']
    optim_sd = None
    best_acc = None
    r_time_avg = None
    if mode == 'train':
        optim_sd = checkpoint["optimizer_state_dict"]
        best_acc = checkpoint["best_acc"]
        r_time_avg = checkpoint["time_per_epoch"]

    # clear out memory
    del checkpoint

    ckpt = {'model_sd': sd,
            'load_epoch': load_epoch,
            'optim_sd': optim_sd,
            'best_acc': best_acc,
            'r_time_avg': r_time_avg}

    return ckpt


class ModelwithInputNormalization(nn.Module):
    def __init__(self, net, mean, std):
        super(ModelwithInputNormalization, self).__init__()
        self.normalizer = helpers.InputNormalize(mean, std)
        self.net = net

    def forward(self,  inp, with_latent=False, fake_relu=False, no_relu=False):
        normalized_inp = self.normalizer(inp)

        if no_relu and (not with_latent):
            print("WARNING: 'no_relu' has no visible effect if 'with_latent is False.")
        if no_relu and fake_relu:
            raise ValueError("Options 'no_relu' and 'fake_relu' are exclusive")

        output = self.net(normalized_inp, with_latent=with_latent,
                          fake_relu=fake_relu, no_relu=no_relu)

        return output



