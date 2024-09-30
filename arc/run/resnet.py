import os

import torch
import torch.nn as nn
import torchvision.models as models

from arc import REPO_ROOT


class ARCResNetClassifier(nn.Module):
    """For multi-label"""

    def __init__(self, num_classes, device):
        super(ARCResNetClassifier, self).__init__()

        # Load a pretrained ResNet
        weights = torch.load(
            os.path.join(
                os.path.dirname(REPO_ROOT),
                "models",
                "resnet_rearc_bcelogits.pth",
            ),
            # weights_only=True,
            # map_location=torch.device("cpu"),
        )
        self.resnet = models.resnet152().to(device)

        # first convolutional layer to accept single-channel input
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=2, padding=3, bias=False
        )

        # replace the final average pooling with adaptive pooling
        self.resnet.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # make it ready for 160-length classification
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)
        self.sigmoid = nn.Sigmoid()

        self.load_custom_state_dict(weights)

    def forward(self, x):
        x = self.resnet(x)
        return self.sigmoid(x)

    def load_custom_state_dict(self, state_dict):
        # remove the "resnet." prefix from keys
        new_state_dict = {
            k.replace("resnet.", ""): v for k, v in state_dict.items()
        }

        # load
        self.resnet.load_state_dict(new_state_dict)
