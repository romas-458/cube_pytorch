import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset


class CubeDataset(Dataset):
    """Cube dataset"""

    def __init__(
            self, x: list, y: list = None, root_dir: str = None, transform=None, is_pil_image: bool = False
    ):
        """
        Args:
            x (list) : list containing path of images
            y (list) : list containing labels corresponding to images
            root_dir (str) : Parent path for reading images
            transform (callabe, Optional): Transforms to be applied
            is_pil_image (bool) : whether list passed to x contains pil images or list of image paths
        """
        self.img_filepath = x
        self.lbls = None if y==None else [0 if el == 'OK' else 1 for el in y]
        self.root_dir = root_dir
        self.transform = transform
        self.is_pil_image = is_pil_image

    def __len__(self):
        return len(self.img_filepath)

    def __getitem__(self, idx):
        if self.root_dir is not None:
            img_filename = os.path.join(self.root_dir, self.img_filepath[idx])
        else:
            img_filename = self.img_filepath[idx]
        if self.is_pil_image:
            img = np.array(img_filename)
        else:
            img = np.array(Image.open(img_filename).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        if self.lbls is not None:
            return img, self.lbls[idx]
        else:
            return img
