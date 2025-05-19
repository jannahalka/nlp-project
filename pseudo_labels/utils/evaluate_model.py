import torch
from typing import Callable, Dict, Optional


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    metrics: Optional[Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]] = None,
) -> Dict[str, float]:
    """
    Evaluate a model on a test dataset.

    Args:
        model: the trained PyTorch model
        dataloader: DataLoader for the test set
        device: torch.device, e.g. torch.device('cuda') or torch.device('cpu')
        criterion: loss function. If None, loss is not computed.
        metrics: dict mapping metric names to functions
            metric_fn(outputs, targets) -> float

    Returns:
        A dict containing:
          - 'loss': average loss over dataset (if criterion given)
          - one entry per metric in `metrics`
    """
    model.eval()
    running_loss = 0.0
    counts = 0
    # initialize metric accumulators
    metric_sums = {name: 0.0 for name in (metrics or {})}

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # loss
            if criterion is not None:
                loss = criterion(outputs, targets)
                running_loss += loss.item() * inputs.size(0)

            # metrics
            if metrics is not None:
                for name, fn in metrics.items():
                    # assume fn takes (outputs, targets) and returns scalar
                    metric_sums[name] += fn(outputs, targets) * inputs.size(0)

            counts += inputs.size(0)

    results: Dict[str, float] = {}
    if criterion is not None:
        results["loss"] = running_loss / counts
    if metrics is not None:
        for name, total in metric_sums.items():
            results[name] = total / counts

    return results

