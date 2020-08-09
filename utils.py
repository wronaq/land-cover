from pathlib import Path
from typing import Dict

import cv2
from torch.utils.data import Dataset


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
