"""Model definitions for knee MRI abnormality detection."""

import torch
import torch.nn as nn
from torchvision.models import resnet18


class KneeClassifier(nn.Module):
    """
    Multi-label knee MRI classifier built on a ResNet-18 encoder.
    """

    def __init__(
        self,
        encoder,
        num_labels=12,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = encoder

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classifier(features)

        return logits


def build_resnet18_encoder():
    """
    Build the ResNet-18 encoder architecture used during SSL.
    """
    encoder = resnet18(
        weights=None
    )

    # Remove ImageNet classification head
    encoder.fc = nn.Identity()

    return encoder


def load_ssl_encoder(
    checkpoint_path,
    map_location="cpu",
):
    """
    Load the pretrained SSL encoder weights.
    """
    encoder = build_resnet18_encoder()

    state_dict = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    encoder.load_state_dict(
        state_dict,
        strict=True,
    )

    return encoder


def build_supervised_model(
    checkpoint_path,
    num_labels=12,
    dropout=0.3,
    device=None,
):
    """
    Load the SSL encoder and attach the supervised classification head.
    """
    encoder = load_ssl_encoder(
        checkpoint_path,
        map_location="cpu",
    )

    model = KneeClassifier(
        encoder=encoder,
        num_labels=num_labels,
        dropout=dropout,
    )

    if device is not None:
        model = model.to(device)

    return model
