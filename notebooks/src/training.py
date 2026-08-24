"""Training and evaluation utilities for knee MRI classification."""

from collections import defaultdict

import torch


def aggregate_study_logits(
    logits,
    labels,
    study_uids,
):
    """
    Aggregate triplet-level logits to study-level logits
    by averaging all triplet predictions for each study.
    """

    study_logit_sum = defaultdict(lambda: None)
    study_count = defaultdict(int)
    study_labels = {}

    for uid, logit, label in zip(
        study_uids,
        logits,
        labels,
    ):

        if study_logit_sum[uid] is None:
            study_logit_sum[uid] = logit.clone()
            study_labels[uid] = label.clone()
        else:
            study_logit_sum[uid] += logit

        study_count[uid] += 1

    aggregated_logits = []
    aggregated_labels = []
    ordered_uids = []

    for uid in study_logit_sum:

        mean_logit = (
            study_logit_sum[uid]
            / study_count[uid]
        )

        aggregated_logits.append(
            mean_logit
        )

        aggregated_labels.append(
            study_labels[uid]
        )

        ordered_uids.append(uid)

    return (
        torch.stack(aggregated_logits),
        torch.stack(aggregated_labels),
        ordered_uids,
    )


@torch.no_grad()
def validate_study_level(
    model,
    loader,
    criterion,
    device,
    use_amp=True,
):
    """
    Run validation and return study-level loss.
    """

    model.eval()

    all_logits = []
    all_labels = []
    all_studies = []

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"]

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                use_amp
                and device.type == "cuda"
            ),
        ):
            logits = model(images)

        all_logits.append(
            logits.detach().cpu()
        )

        all_labels.append(
            labels.detach().cpu()
        )

        all_studies.extend(
            batch["study_uid"]
        )

    all_logits = torch.cat(
        all_logits
    )

    all_labels = torch.cat(
        all_labels
    )

    (
        study_logits,
        study_labels,
        study_uids,
    ) = aggregate_study_logits(
        all_logits,
        all_labels,
        all_studies,
    )

    loss = criterion(
        study_logits.to(device),
        study_labels.to(device),
    )

    return {
        "loss": float(loss.item()),
        "logits": study_logits,
        "labels": study_labels,
        "study_uids": study_uids,
    }


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    use_amp=True,
    max_grad_norm=1.0,
):
    """
    Train the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    samples_seen = 0

    for batch in loader:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                use_amp
                and device.type == "cuda"
            ),
        ):

            logits = model(images)

            loss = criterion(
                logits,
                labels,
            )

        if scaler is not None:

            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            scaler.step(
                optimizer
            )

            scaler.update()

        else:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            optimizer.step()

        batch_size = images.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        samples_seen += batch_size

    return (
        running_loss
        / samples_seen
    )
