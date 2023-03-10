# Copyright 2021-2023 Lawrence Livermore National Security, LLC and other
# MuyGPyS Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

import MuyGPyS._src.math.torch as torch


def _cross_entropy_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
) -> torch.ndarray:
    loss = torch.nn.CrossEntropyLoss()
    one_hot_targets = torch.where(targets > 0.0, 1.0, 0.0)
    softmax_predictions = predictions.softmax(dim=1)

    return (
        -torch.mean(
            one_hot_targets * torch.log(softmax_predictions)
            + (1 - one_hot_targets) * torch.log(1 - softmax_predictions)
        )
        * one_hot_targets.shape[0]
    )


def _mse_fn_unnormalized(
    predictions: torch.ndarray,
    targets: torch.ndarray,
) -> float:
    return torch.sum((predictions - targets) ** 2)


def _mse_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
) -> float:
    batch_count, response_count = predictions.shape
    return _mse_fn_unnormalized(predictions, targets) / (
        batch_count * response_count
    )


def _lool_fn(
    predictions: torch.ndarray,
    targets: torch.ndarray,
    variances: torch.ndarray,
    sigma_sq: torch.ndarray,
) -> float:
    scaled_variances = torch.unsqueeze(variances * sigma_sq, dim=1)
    return torch.sum(
        torch.div((predictions - targets) ** 2, scaled_variances)
        + torch.log(scaled_variances)
    )
