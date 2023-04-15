import torch


def binary_cross_entropy_cls(predictions: torch.Tensor, labels: torch.Tensor):
    """
    https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    Parameters
    ----------
    predictions: (B, ) must be in [0, 1]
    labels: (B, )
    size_average
    check_input

    Returns
    -------

    """
    assert predictions.size() == labels.size()
    criterion = torch.nn.BCELoss()  # should I create new instance here!!!!
    return criterion(predictions, labels.float())


def cross_entroy(predictions: torch.Tensor, labels: torch.tensor):
    assert predictions.shape[0] == labels.shape[0]
    criterion = torch.nn.CrossEntropyLoss()
    return criterion(predictions, labels.long())
