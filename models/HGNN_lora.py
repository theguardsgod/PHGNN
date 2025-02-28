from torch import nn
from models import HGNN_conv_lora
import torch.nn.functional as F


class HGNN_lora(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN_lora, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv_lora(in_ch, n_hid)
        self.hgc2 = HGNN_conv_lora(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x
