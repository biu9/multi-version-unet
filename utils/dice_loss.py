import torch.nn as nn
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        #y_pred = y_pred[:, 0].contiguous().view(-1)
        #y_true = y_true[:, 0].contiguous().view(-1)
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        #print('y_pred.sum()', y_pred.sum())
        #print('y_true.sum()', y_true.sum())
        intersection = (y_pred * y_true).sum()
        #print('intersection', intersection)
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc