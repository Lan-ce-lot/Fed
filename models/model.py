import torch
from torch import nn


class LocalModel(nn.Module):
    def __init__(self, base, head):
        super(LocalModel, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out


class ClientModel(nn.Module):
    def __init__(self, base, g_fea, p_fea, head):
        super(ClientModel, self).__init__()

        self.base = base
        self.g_fea = g_fea
        self.p_fea = p_fea
        self.head = head
    def forward(self, x):
        out = self.base(x)
        gout = self.g_fea(out)
        pout = self.p_fea(out)
        out = gout.detach()+pout
        out = self.head(out)
        return out
