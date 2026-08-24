"""MRI preprocessing utilities for knee abnormality detection."""

from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as F


TARGET_SIZE = 224
TARGET_SPACING = 0.5


def physical_slice_position(ds):
    """
    Calculate a DICOM slice's physical position.

    Uses ImageOrientationPatient and ImagePositionPatient when available,
    with SliceLocation and InstanceNumber as fallbacks.
    """
    try:
        orientation = np.asarray(
            ds.ImageOrientationPatient,
            dtype=float,
        )

        position = np.asarray(
            ds.ImagePositionPatient,
            dtype=float,
        )

        row = orientation[:3]
        col = orientation[3:]

        normal = np.cross(row, col)

        return float(
            np.dot(position, normal)
        )

    except Exception:
        try:
            return float(ds.SliceLocation)

        except Exception:
            return float(ds.InstanceNumber)


def load_dicom_pixels(path):
    """
    Load a DICOM image and apply rescale slope/intercept.
    """
    ds = pydicom.dcmread(
        str(path),
        force=True,
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    image = (
        image * slope
        + intercept
    )

    # Correct MONOCHROME1 images
    if getattr(
        ds,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":

        image = (
            image.max()
            + image.min()
            - image
        )

    return image


def get_pixel_spacing(ds):
    """
    Return row and column pixel spacing.

    Falls back to the target spacing when metadata is unavailable.
    """
    try:
        spacing = ds.PixelSpacing

        row_spacing = float(spacing[0])
        col_spacing = float(spacing[1])

    except Exception:
        row_spacing = TARGET_SPACING
        col_spacing = TARGET_SPACING

    return row_spacing, col_spacing


def normalize_image(
    image,
    p01,
    p99,
):
    """
    Clip image intensities to the 1st-99th percentile
    range and scale to [0, 1].
    """
    image = np.clip(
        image,
        p01,
        p99,
    )

    image = (
        (image - p01)
        / (p99 - p01 + 1e-6)
    )

    return image.astype(
        np.float32
    )


def resize_to_spacing(
    image,
    row_spacing,
    col_spacing,
    target_spacing=TARGET_SPACING,
):
    """
    Resize an MRI slice to approximately uniform physical spacing.
    """
    h, w = image.shape

    new_h = max(
        1,
        int(
            round(
                h
                * row_spacing
                / target_spacing
            )
        ),
    )

    new_w = max(
        1,
        int(
            round(
                w
                * col_spacing
                / target_spacing
            )
        ),
    )

    image = torch.from_numpy(
        image
    ).float()[None, None]

    image = F.interpolate(
        image,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )

    return image[0, 0]


def centre_crop_or_pad(
    image,
    size=TARGET_SIZE,
):
    """
    Centre crop or zero-pad an image to a fixed square size.
    """
    h, w = image.shape

    pad_h = max(
        0,
        size - h,
    )

    pad_w = max(
        0,
        size - w,
    )

    if pad_h > 0 or pad_w > 0:

        top = pad_h // 2
        bottom = pad_h - top

        left = pad_w // 2
        right = pad_w - left

        image = F.pad(
            image,
            (
                left,
                right,
                top,
                bottom,
            ),
        )

    h, w = image.shape

    y0 = (h - size) // 2
    x0 = (w - size) // 2

    return image[
        y0:y0 + size,
        x0:x0 + size,
    ]


def preprocess_slice(
    path,
    p01,
    p99,
):
    """
    Apply the complete preprocessing pipeline to one DICOM slice.

    Returns
    -------
    torch.Tensor
        Tensor with shape [224, 224].
    """
    ds = pydicom.dcmread(
        str(path),
        force=True,
    )

    image = ds.pixel_array.astype(
        np.float32
    )

    slope = float(
        getattr(ds, "RescaleSlope", 1.0)
    )

    intercept = float(
        getattr(ds, "RescaleIntercept", 0.0)
    )

    image = (
        image * slope
        + intercept
    )

    if getattr(
        ds,
        "PhotometricInterpretation",
        "",
    ) == "MONOCHROME1":

        image = (
            image.max()
            + image.min()
            - image
        )

    row_spacing, col_spacing = (
        get_pixel_spacing(ds)
    )

    image = normalize_image(
        image,
        p01,
        p99,
    )

    image = resize_to_spacing(
        image,
        row_spacing,
        col_spacing,
    )

    image = centre_crop_or_pad(
        image,
        TARGET_SIZE,
    )

    return image


def build_2_5d_triplet(
    previous_path,
    centre_path,
    next_path,
    p01,
    p99,
):
    """
    Construct a previous-centre-next 2.5D MRI tensor.

    Returns
    -------
    torch.Tensor
        Tensor with shape [3, 224, 224].
    """
    previous = preprocess_slice(
        previous_path,
        p01,
        p99,
    )

    centre = preprocess_slice(
        centre_path,
        p01,
        p99,
    )

    next_slice = preprocess_slice(
        next_path,
        p01,
        p99,
    )

    return torch.stack(
        [
            previous,
            centre,
            next_slice,
        ],
        dim=0,
    )
