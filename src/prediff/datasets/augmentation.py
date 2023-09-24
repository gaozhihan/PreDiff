from typing import Sequence
import random
from torch import nn
import torchvision.transforms.functional as TF


class TransformsFixRotation(nn.Module):
    r"""
    Rotate by one of the given angles.

    Example: `rotation_transform = MyRotationTransform(angles=[-30, -15, 0, 15, 30])`
    """

    def __init__(self, angles):
        super(TransformsFixRotation, self).__init__()
        if not isinstance(angles, Sequence):
            angles = [angles, ]
        self.angles = angles

    def forward(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(angles={self.angles})"
