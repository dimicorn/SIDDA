from typing import Callable, Optional
import os
from glob import glob
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pre.load import fits2numpy
from pre.preprocess import preprocess_lognorm


class Shapes(Dataset):
    """Dataset class for the shapes dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class AstroObjects(Dataset):
    """Dataset class for the astronomical objects dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class MnistM(Dataset):
    """Dataset class for the MNIST-M dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class GZEvo(Dataset):
    """Dataset class for the Galaxy Zoo Evolution dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length
    
    
class MRSSC2(Dataset):
    """Dataset class for the Galaxy Zoo Evolution dataset.

    Args:
        input_path (str): Path to the input data.
        output_path (Optional[str], optional): Path to the output data. Defaults to None.
        transform (Optional[Callable], optional): Transform to apply to the data. Defaults to None.
        target_domain (bool, optional): Whether the dataset is used for target domain. Defaults to False.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        self.target_domain = target_domain

        try:
            self.img = np.load(self.input_path)
            if not self.target_domain and self.output_path is not None:
                self.label = np.load(self.output_path)
        except Exception as e:
            raise RuntimeError(
                f"Error loading data from {input_path} and {output_path}: {e}"
            )

        if (
            not self.target_domain
            and self.output_path is not None
            and len(self.img) != len(self.label)
        ):
            raise ValueError("Input and output files must have the same length.")

        self.length = len(self.img)

    def __getitem__(self, idx: int):
        img = self.img[idx]
        if self.transform:
            img = self.transform(img)

        if self.target_domain:
            return img

        label = torch.tensor(self.label[idx], dtype=torch.long)
        return img, label

    def __len__(self) -> int:
        return self.length


class Astrogeo(Dataset):
    def __init__(
        self,
        input_path,
        output_path: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_domain: bool = False
    ):
        self.transform = transform
        self.target_domain = target_domain

        if target_domain:
            self.files = self._scan(input_path)
        else:
            self.samples = []

            classes = sorted(os.listdir(input_path))
            self.class_to_idx = {c: i for i, c in enumerate(classes)}

            for cls in classes:
                cls_path = os.path.join(input_path, cls)
                for f in glob(os.path.join(cls_path, "*.fits")):
                    self.samples.append((f, self.class_to_idx[cls]))

            self.targets = [label for _, label in self.samples]
            # print(self.samples[:5])

    def _scan(self, root_dir: str, json_file: str = "/home/zagorulia/ml/scripts/filenames.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            files = json.load(f)
        files = [
            os.path.join(root_dir, file.split("_")[0], file) for file in files
        ]
        files.sort()
        return files

    def __len__(self):
        return len(self.files) if self.target_domain else len(self.samples)

    def __getitem__(self, idx):
        if self.target_domain:
            path = self.files[idx]
        else:
            path, label = self.samples[idx]
        data_raw = fits2numpy(path)
        data = preprocess_lognorm(data_raw)
        x = torch.from_numpy(data).unsqueeze(0)

        if self.transform is not None:
            x = self.transform(x)

        if self.target_domain:
            return x
    
        return x, label


dataset_dict = {
    "shapes": Shapes,
    "astro_objects": AstroObjects,
    "mnist_m": MnistM,
    "gz_evo": GZEvo,
    "mrssc2": MRSSC2, 
    "astrogeo": Astrogeo,
}

gz_evo_classes = (
    "barred_spiral",
    "edge_on_disk",
    "featured_without_bar_or_spiral",
    "smooth_cigar",
    "smooth_round",
    "unbarred_spiral",
)
mnist_m_classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
shapes_classes = ("line", "rectangle", "circle")
astro_objects_classes = ("elliptical", "spiral", "stars")
mrssc2_classes = ("city", "coast", "desert", "farmland", "lake", "mountain", "river")
astrogeo_classes = ("0", "1")

classes_dict = {
    "shapes": shapes_classes,
    "astro_objects": astro_objects_classes,
    "mnist_m": mnist_m_classes,
    "gz_evo": gz_evo_classes,
    "mrssc2": mrssc2_classes,
    "astrogeo": astrogeo_classes,
}
