import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = self._make_layers()
        self.classifier = nn.Linear(512, num_classes)

    def _make_layers(self):
        layers = [nn.Conv2d(3, 32, 3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(32, 32, 3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Conv2d(32, 64, 3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(64, 64, 3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Conv2d(64, 128, 3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(128, 128, 3, padding=1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(2, 2)]

        layers += [nn.Flatten()]
        layers += [nn.Linear(128 * 4 * 4, 512)]
        layers += [nn.ReLU(inplace=True)]

        return nn.Sequential(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        assert (not fake_relu) and (not no_relu),  \
            "fake_relu and no_relu not yet supported for this architecture"
        latent = self.features(x)
        out = self.classifier(latent)
        if with_latent:
            return out, latent
        return out


dnn = DNN

#def test():
#
#    net = DNN()
#    o, l = net(torch.randn(16,3,32,32), with_latent=True)
#    print(o.shape, l.shape)
#
#test()
