import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import numpy as np
import random
import torch


class EvaluationMonitor:
    def __init__(self):
        self.total_images = 0
        self.evaluated_images = 0
        self.evaluation_running = True

    def reset(self):
        self.__init__()

    def get_progress(self):
        if self.total_images == 0:
            return 0.0
        return round((self.evaluated_images / self.total_images), 2)


class Monitor:
    def __init__(self):
        self.n_epochs = 0
        self.batches_per_epoch = 0
        self.current_epoch = 0
        self.current_batch = 0

    def reset(self):
        self.__init__()

    def get_progress(self):
        # TODO smoother progress
        if self.n_epochs == 0:
            return 0.0
        batches_per_epoch = max(1, self.batches_per_epoch)
        total_batches = batches_per_epoch * self.n_epochs
        current_total_batch = (batches_per_epoch * self.current_epoch) + self.current_batch
        return round((current_total_batch / total_batches), 2)


class Terminator:
    def __init__(self):
        self.terminate_flag = False

    def reset(self):
        self.__init__()

    def terminate(self, *args, **kwargs):
        # args are for campian interface compatibility
        self.terminate_flag = True


# strong augmentations
def get_train_transforms(height, width, means, stds):
    """
    Apply training transformations from albumentation library
    """
    trn_transform = A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST, p=1),
            A.HorizontalFlip(p=0.5),
            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),
            A.ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0, value=0,
                               mask_value=0),
            A.OneOf(
                [
                    A.RandomBrightness(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.IAASharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.5,
            ),
            A.Normalize(mean=means, std=stds, p=1),
            ToTensor(),
        ], p=1
    )
    return trn_transform


def get_val_transforms(height, width, means, stds):
    """
    Apply val transformations from albumentation library
    """
    val_transform = A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST, p=1),
            A.Normalize(mean=means, std=stds, p=1),
            ToTensor(),
        ], p=1
    )
    return val_transform


def set_global_seeds():
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    random.seed(hash("setting random seeds") % 2 ** 32 - 1)
    np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
    torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)
