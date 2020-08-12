from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter


class LandCoverDataset(Dataset):
    """Land Cover Dataset."""

    def __init__(self, root_dir: str, transform=None) -> None:
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path.cwd() / root_dir
        self._images_dir = self.root_dir / "images"
        self._labels_dir = self.root_dir / "labels"
        self._images_list = [
            x for x in self._images_dir.iterdir() if x.suffix == ".tif"
        ]
        self._labels_list = [
            x for x in self._labels_dir.iterdir() if x.suffix == ".tif"
        ]
        self.transform = transform

        assert len(self._images_list) == len(
            self._labels_list
        ), "Different number of images and labels!"

        self._images_list.sort()
        self._labels_list.sort()
        assert [x.stem.split(".")[1] for x in self._images_list] == [
            x.stem.split(".")[1] for x in self._labels_list
        ], "Different names for images and labels!"

    def __len__(self) -> int:
        return len(self._images_list)

    def __getitem__(self, index: int) -> Dict:
        image_file = self._images_list[index]
        label_file = self._labels_list[index]

        image = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)
        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample


####
# transformations
####


class ToTensor:
    """Transform to tensor."""

    def __call__(self, sample: Dict) -> Dict:
        assert (
            "image" in sample.keys() and "label" in sample.keys()
        ), "Wrong sample format!"

        image, label = sample["image"], sample["label"]

        # add channel to grayscale image
        label = label.reshape((label.shape[0], label.shape[1], 1))

        # move channel to first dimmention
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        return {"image": torch.from_numpy(image), "label": torch.from_numpy(label)}


class Resize:
    """Resize input image to desired output size.

    Args:
        output_size (int or tuple): Desired output size. If int then smaller of image edges
        takes this size keeping image aspect ratio.
    """

    def __init__(self, output_size: Union[int, Tuple]) -> None:
        self.output_size = output_size

        assert isinstance(self.output_size, (int, tuple)), "Wrong output size format!"

    def __call__(self, sample: Dict) -> Dict:
        assert (
            "image" in sample.keys() and "label" in sample.keys()
        ), "Wrong sample format!"
        assert (
            sample["image"].shape[:2] == sample["label"].shape[:2]
        ), "Different size of image and label!"

        image, label = sample["image"], sample["label"]
        h, w = image.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            elif h < w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size, self.output_size
        elif isinstance(self.output_size, tuple):
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        image = cv2.resize(image, (new_h, new_w))
        label = cv2.resize(label, (new_h, new_w))

        return {"image": image, "label": label}


class Normalize:
    """Normalize images.

    Args:
        mean (list): List of means for every channel.
        std (list): List of standard deviations for every channel.
    """

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample["image"], sample["label"]

        assert (
            len(self.mean) == image.shape[2] and len(self.std) == image.shape[2]
        ), "Wrong mean or std size!"

        params = {"mean": {}, "std": {}}
        for i, (mean, std) in enumerate(zip(self.mean, self.std)):
            params["mean"][i] = np.ones((image.shape[0], image.shape[1], 1)) * mean
            params["std"][i] = np.ones((image.shape[0], image.shape[1], 1)) * std

        means = np.concatenate(
            (params["mean"][0], params["mean"][1], params["mean"][2]), axis=2
        )
        stds = np.concatenate(
            (params["std"][0], params["std"][1], params["std"][2]), axis=2
        )
        image = (image - means) / stds

        return {"image": image, "label": label}


class BrightnessJitter(ColorJitter):
    """Randomly change the brightness of image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
    """

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample["image"], sample["label"]

        if self.brightness:
            brightness_factor = (
                torch.tensor(1.0)
                .uniform_(self.brightness[0], self.brightness[1])
                .item()
            )
            image = F.adjust_brightness(image, brightness_factor)

        return {"image": image, "label": label}


class ContrastJitter(ColorJitter):
    """Randomly change the contrast of image.

    Args:
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
    """

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample["image"], sample["label"]

        if self.contrast:
            contrast_factor = (
                torch.tensor(1.0).uniform_(self.contrast[0], self.contrast[1]).item()
            )
            image = F.adjust_contrast(image, contrast_factor)

        return {"image": image, "label": label}


class SaturationJitter(ColorJitter):
    """Randomly change the saturation of image.

    Args:
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
    """

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample["image"], sample["label"]

        if self.saturation:
            saturation_factor = (
                torch.tensor(1.0)
                .uniform_(self.saturation[0], self.saturation[1])
                .item()
            )
            image = F.adjust_saturation(image, saturation_factor)

        return {"image": image, "label": label}


class HueJitter(ColorJitter):
    """Randomly change the hue of image.

    Args:
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __call__(self, sample: Dict) -> Dict:
        image, label = sample["image"], sample["label"]

        if self.hue:
            hue_factor = torch.tensor(1.0).uniform_(self.hue[0], self.hue[1]).item()
            image = F.adjust_hue(image, hue_factor)

        return {"image": image, "label": label}
