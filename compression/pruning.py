import torch
import torch.nn as nn


def global_magnitude_prune(model, prune_ratio=0.2):
    """
    Global unstructured magnitude pruning.

    Finds one global threshold across all Conv2d / Linear weights,
    then sets the smallest-magnitude prune_ratio fraction to zero.

    Example:
        prune_ratio = 0.2
        -> approximately 20% of eligible weights become zero.

    Returns:
        model
        stats dictionary
    """

    if not (0.0 <= prune_ratio < 1.0):
        raise ValueError("prune_ratio must be in [0, 1)")

    # --------------------------------------------------------
    # Collect all Conv2d / Linear weights
    # --------------------------------------------------------

    all_weights = []

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            if module.weight is not None:
                all_weights.append(
                    module.weight.data.abs().flatten()
                )

    if len(all_weights) == 0:
        raise RuntimeError(
            "No Conv2d / Linear weights found."
        )

    all_weights = torch.cat(all_weights)

    total_weights = all_weights.numel()

    # --------------------------------------------------------
    # Number of weights to prune
    # --------------------------------------------------------

    num_to_prune = int(
        prune_ratio * total_weights
    )

    # Nothing to prune
    if num_to_prune == 0:

        return model, {
            "total_weights": total_weights,
            "pruned_weights": 0,
            "remaining_weights": total_weights,
            "actual_sparsity": 0.0,
            "threshold": 0.0
        }

    # --------------------------------------------------------
    # Global threshold
    # --------------------------------------------------------

    # kthvalue is 1-indexed
    threshold = torch.kthvalue(
        all_weights,
        num_to_prune
    ).values

    # --------------------------------------------------------
    # Apply pruning
    # --------------------------------------------------------

    pruned_weights = 0

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            if module.weight is None:
                continue

            weight = module.weight.data

            mask = (
                weight.abs() > threshold
            )

            # Count zeros introduced
            pruned_weights += (
                (~mask).sum().item()
            )

            module.weight.data.mul_(mask)

    remaining_weights = (
        total_weights - pruned_weights
    )

    actual_sparsity = (
        pruned_weights / total_weights
    )

    stats = {
        "total_weights": total_weights,
        "pruned_weights": pruned_weights,
        "remaining_weights": remaining_weights,
        "actual_sparsity": actual_sparsity,
        "threshold": threshold.item()
    }

    return model, stats