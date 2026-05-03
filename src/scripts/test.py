import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import yaml
from dataset import classes_dict, dataset_dict
from models import model_dict
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
import torchvision.transforms.v2 as v2 
from tqdm import tqdm
from pre.load import fits2numpy
from pre.preprocess import preprocess_lognorm


class FITSFolder(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []

        classes = sorted(os.listdir(root))
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_path = os.path.join(root, cls)
            for f in glob(os.path.join(cls_path, "*.fits")):
                self.samples.append((f, self.class_to_idx[cls]))

        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data_raw = fits2numpy(path)
        data = preprocess_lognorm(data_raw)
        # data_lognorm = preprocess_lognorm(data_raw)
        # data = np.stack([data_pre, data_lognorm], axis=0)

        if self.transform:
            img = Image.fromarray(data, mode="F")
            x = self.transform(img)
        else:
            x = torch.from_numpy(data).unsqueeze(0)

        return x, label


class FilterAndRemap(Dataset):
    """
    Keeps only samples whose original label is in `keep`.
    Remaps original labels -> {0,1} using `remap` dict.
    """

    def __init__(self, base_ds, keep, remap):
        self.base_ds = base_ds
        self.keep = set(keep)
        self.remap = remap

        # works for ImageFolder / datasets with .targets
        targets = getattr(base_ds, "targets")
        self.indices = [i for i, y in enumerate(targets) if y in self.keep]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base_ds[self.indices[i]]
        y = self.remap[y]
        return x, y

def adapt_resnet_to_1ch(model):
    old_conv = model.conv1

    new_conv = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    # convert RGB → mono by averaging filters
    with torch.no_grad():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    model.conv1 = new_conv
    return model


class ResNet18Custom(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        self.backbone = adapt_resnet_to_1ch(self.backbone)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        latent_space = self.backbone(x)   # shape: [B, 512]
        logits = self.fc(latent_space)    # shape: [B, num_classes]
        return latent_space, logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(directory_path: str, 
                model_name: str,
                dataset_name: str) -> list:
    """Load models from a directory

    Args:
        directory_path (str): directory with the trained models
        model_name (str): name of the model to be loaded (following the model_dict)

    Returns:
        loaded models (list): list of loaded models
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Loading {model_name} from {file_path}...")
            model = ResNet18Custom()
            model.eval()
            model.load_state_dict(torch.load(file_path, map_location=device))

            model_name_no_ext = file_name[:-3]
            models.append((model, model_name_no_ext))
            print(f"Finished Loading {model_name} from {file_path}")

    if not models:
        print(
            f"No models containing 'best_model' ending with '.pt' found in {directory_path}."
        )

    return models


@torch.no_grad()
def compute_metrics(
    test_loader: DataLoader,
    model: nn.Module,
    model_name: str,
    save_dir: str,
    output_name: str,
    classes: tuple,
):
    """Compute metrics for the model

    Args:
        test_loader (nn.DataLoader): test data loader
        model (nn.Module): model to be evaluated
        model_name (str): name of the model
        save_dir (str): directory to save the results
        output_name (str): name of the output file
        classes (tuple): classes to be evaluated

    Returns:
        sklearn_report (dict): sklearn classification report
    """

    y_pred, y_true, feature_maps = [], [], []
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, output = batch
        input, _ = input.to(device).float(), output.to(device)
        features, preds = model(input)
        _, predicted_class = torch.max(preds.data, 1)
        feature_maps.extend(features.cpu().numpy())

        y_pred.extend(predicted_class.cpu().numpy())
        y_true.extend(output.cpu().numpy())

    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)
    features_dir = os.path.join(save_dir, "latent_vectors")
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    y_pred_dir = os.path.join(save_dir, "y_pred")
    if not os.path.exists(y_pred_dir):
        os.makedirs(y_pred_dir)
    np.save(
        f"{features_dir}/latent_vecs_{model_name}_{output_name}.npy", flattened_features
    )
    np.save(f"{y_pred_dir}/y_pred_{model_name}_{output_name}.npy", y_pred)

    confusion_matrix_dir = os.path.join(save_dir, "confusion_matrix")
    if not os.path.exists(confusion_matrix_dir):
        os.makedirs(confusion_matrix_dir)

    sklearn_report = classification_report(
        y_true, y_pred, output_dict=True, target_names=classes
    )

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(
        os.path.join(
            confusion_matrix_dir, f"confusion_matrix_{model_name}_{output_name}.png"
        ),
        bbox_inches="tight",
    )
    plt.close()

    return sklearn_report


@torch.no_grad()
def main(
    model_dir: str,
    output_name: str,
    x_test_path: str,
    y_test_path: str,
    model_name: str,
    classes: tuple,
    dataset: str,
):
    metrics_dir = os.path.join(model_dir, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    if dataset in ["shapes", "astronomical_objects"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                transforms.Resize(100),
            ]
        )

    elif dataset == "mnist_m":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.Resize(32),
            ]
        )

    elif dataset == "gz_evo":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.Resize(100),
            ]
        )
        
    elif dataset == "mrssc2":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(256),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
    elif dataset == "astrogeo":
        transform = v2.ToDtype(torch.float32, scale=False)

    # test_dataset = dataset_dict[dataset](x_test_path, y_test_path, transform=transform, target_domain=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    # test_dataset = FITSFolder("/data/zagorulia/val_fits")
    # c_idx = test_dataset.class_to_idx["0"]
    # d_idx = test_dataset.class_to_idx["2"]
    # final_test_ds = FilterAndRemap(
    #     base_ds=test_dataset,
    #     keep=[c_idx, d_idx],
    #     remap={c_idx: 0, d_idx: 1},
    # )
    # filtered_class_names = ["point", "jet"]
    # test_dataloader = DataLoader(final_test_ds, batch_size=128, shuffle=True)

    test_dataset = FITSFolder("/data/zagorulia/synt_14_02_26")
    filtered_class_names = ["point", "jet"]
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    models = load_models(model_dir, model_name, dataset)
    if not models:
        print("Models could not be loaded.")
        return

    for model, model_file_name in models:
        model_metrics = {
            class_name: {"precision": [], "recall": [], "f1-score": [], "support": []}
            for class_name in filtered_class_names
        }
        model_metrics["accuracy"] = []
        model_metrics["macro avg"] = {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": [],
        }
        model_metrics["weighted avg"] = {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": [],
        }

        full_report = compute_metrics(
            test_loader=test_dataloader,
            model=model,
            model_name=model_name,
            save_dir=model_dir,
            output_name=f"{output_name}_{model_file_name}",
            classes=filtered_class_names,
        )
        model_metrics = full_report

        print("Compiling Metrics")
        output_file_name = f"{output_name}_{model_file_name}.yaml"
        with open(os.path.join(metrics_dir, output_file_name), "w") as file:
            yaml.dump(model_metrics, file)

        print(f"Metrics saved at {os.path.join(model_dir, output_file_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gz_evo",
        help="Dataset to be used for evaluation",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained models"
    )
    parser.add_argument(
        "--x_test_path", type=str, required=False, help="Path to the x_test data"
    )
    parser.add_argument(
        "--y_test_path", type=str, required=False, help="Path to the y_test data"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Name of the output file for the results",
    )
    parser.add_argument(
        "--model_name", type=str, help="Name of the model to be evaluated"
    )
    
    args = parser.parse_args()

    main(
        model_dir=args.model_path,
        output_name=args.output_name,
        x_test_path=args.x_test_path,
        y_test_path=args.y_test_path,
        model_name=args.model_name,
        classes=classes_dict[args.dataset],
        dataset = args.dataset
    )
