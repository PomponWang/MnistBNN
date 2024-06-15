import torch.nn as nn
from .. import functional as BF

class BKLLoss(nn.Module):
    """
    Loss for calculating KL divergence of baysian neural network model.

    Arguments:
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be averaged over all parameters.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.    
    """
    __constants__ = ['reduction']

    def __init__(self, reduction='mean', last_layer_only=False):
        super().__init__()
        self.reduction = reduction
        self.last_layer_only = last_layer_only

    def forward(self, model):
        return BF.bayesian_kl_loss(model, reduction=self.reduction, last_layer_only=self.last_layer_only)