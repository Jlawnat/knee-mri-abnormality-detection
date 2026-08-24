"""PyTorch datasets for knee MRI abnormality detection."""

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import build_2_5d_triplet


def supervised_augmentation(x):
    """
    Mild MRI-safe augmentation.

    Parameters
    ----------
    x : torch.Tensor
        2.5D MRI tensor with shape [3, 224, 224].
    """
    # Horizontal flip
    if torch.rand(1).item() < 0.5:
        x = torch.flip(
            x,
            dims=[2],
        )

    # Small contrast adjustment
    contrast = (
        0.9
        + torch.rand(1).item() * 0.2
    )

    mean = x.mean()

    x = (
        (x - mean) * contrast
        + mean
    )

    # Small brightness adjustment
    brightness = (
        torch.rand(1).item() * 0.08
        - 0.04
    )

    x = x + brightness

    # Light Gaussian noise
    if torch.rand(1).item() < 0.3:

        noise_sigma = (
            torch.rand(1).item()
            * 0.015
        )

        x = (
            x
            + torch.randn_like(x)
            * noise_sigma
        )

    return x.clamp(
        0.0,
        1.0,
    )


class SupervisedKneeDataset(Dataset):
    """
    Dataset for labelled 2.5D knee MRI triplets.
    """

    def __init__(
        self,
        dataframe,
        label_cols,
        augment=False,
    ):
        self.df = (
            dataframe
            .reset_index(drop=True)
        )

        self.label_cols = list(label_cols)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = build_2_5d_triplet(
            previous_path=row["PreviousPath"],
            centre_path=row["CentrePath"],
            next_path=row["NextPath"],
            p01=float(row["P01"]),
            p99=float(row["P99"]),
        )

        if self.augment:
            image = supervised_augmentation(
                image
            )

        labels = torch.tensor(
            row[self.label_cols]
            .to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "labels": labels,
            "study_uid": row["StudyInstanceUID"],
            "series_uid": row["SeriesInstanceUID"],
            "plane": row["Anatomical_Plane"],
        }


class TestKneeDataset(Dataset):
    """
    Dataset for unlabelled test MRI triplets.
    """

    def __init__(
        self,
        dataframe,
    ):
        self.df = (
            dataframe
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = build_2_5d_triplet(
            previous_path=row["PreviousPath"],
            centre_path=row["CentrePath"],
            next_path=row["NextPath"],
            p01=float(row["P01"]),
            p99=float(row["P99"]),
        )

        return {
            "image": image,
            "study_uid": row["StudyInstanceUID"],
            "series_uid": row["SeriesInstanceUID"],
            "plane": row["Anatomical_Plane"],
        }
