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
