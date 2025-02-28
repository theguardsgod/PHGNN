from torch import nn
from models import HGNN_conv
import torch.nn.functional as F
import torch

class HGNN_Adapter(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN_Adapter, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

        bottleneck_dim = 15
        prompt_num = 2
        num_layer = 2
        gating = 0.01
        emb_dim = n_hid
        self.gating_parameter = torch.nn.Parameter(torch.zeros(prompt_num, num_layer, 1))
        self.gating_parameter.data += gating
        self.register_parameter('gating_parameter', self.gating_parameter)
        self.gating = self.gating_parameter

        self.prompts = torch.nn.ModuleList()
        for i in range(prompt_num):
            self.prompts.append(torch.nn.ModuleList())

        for layer in range(num_layer):
            for i in range(prompt_num):
                if layer == 0 and i == 0:
                    if bottleneck_dim > 0:
                        self.prompts[i].append(torch.nn.Sequential(
                            torch.nn.Linear(in_ch if i > 0 else in_ch, bottleneck_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(bottleneck_dim, emb_dim),
                            torch.nn.BatchNorm1d(emb_dim)
                        ))
                        torch.nn.init.zeros_(self.prompts[i][-1][2].weight.data)
                        torch.nn.init.zeros_(self.prompts[i][-1][2].bias.data)
                    else:
                        self.prompts[i].append(torch.nn.BatchNorm1d(emb_dim))
                elif layer == 1:
                    if i == 0:
                        if bottleneck_dim > 0:
                            self.prompts[i].append(torch.nn.Sequential(
                                torch.nn.Linear(emb_dim if i > 0 else emb_dim, bottleneck_dim),
                                torch.nn.ReLU(),
                                torch.nn.Linear(bottleneck_dim, n_class),
                                torch.nn.BatchNorm1d(n_class)
                            ))
                            torch.nn.init.zeros_(self.prompts[i][-1][2].weight.data)
                            torch.nn.init.zeros_(self.prompts[i][-1][2].bias.data)
                        else:
                            self.prompts[i].append(torch.nn.BatchNorm1d(emb_dim))
                    else:
                        if bottleneck_dim > 0:
                            self.prompts[i].append(torch.nn.Sequential(
                                torch.nn.Linear(n_class if i > 0 else n_class, bottleneck_dim),
                                torch.nn.ReLU(),
                                torch.nn.Linear(bottleneck_dim, n_class),
                                torch.nn.BatchNorm1d(n_class)
                            ))
                            torch.nn.init.zeros_(self.prompts[i][-1][2].weight.data)
                            torch.nn.init.zeros_(self.prompts[i][-1][2].bias.data)
                        else:
                            self.prompts[i].append(torch.nn.BatchNorm1d(emb_dim))
                    
                else:
                    if bottleneck_dim > 0:
                        self.prompts[i].append(torch.nn.Sequential(
                            torch.nn.Linear(emb_dim if i > 0 else emb_dim, bottleneck_dim),
                            torch.nn.ReLU(),
                            torch.nn.Linear(bottleneck_dim, emb_dim),
                            torch.nn.BatchNorm1d(emb_dim)
                        ))
                        torch.nn.init.zeros_(self.prompts[i][-1][2].weight.data)
                        torch.nn.init.zeros_(self.prompts[i][-1][2].bias.data)
                    else:
                        self.prompts[i].append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, G):
        delta1 = self.prompts[0][0](x)
        h = self.hgc1(x, G)
        delta2 = self.prompts[1][0](h)
        
        h = h + delta1 * self.gating[0][0]
        h = h + delta2 * self.gating[1][0]

        x = F.relu(h)
        x = F.dropout(x, self.dropout)

        delta1 = self.prompts[0][1](x)
        h = self.hgc2(x, G)
        delta2 = self.prompts[1][1](h)
        
        h = h + delta1 * self.gating[0][1]
        h = h + delta2 * self.gating[1][1]
        
        return h
