import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F

from sklearn.preprocessing import label_binarize
from sklearn import metrics

from algorithms.client.client import Client


class clientFedours(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5, momentum=0.9)

        self.pred = copy.deepcopy(self.model.head)
        self.opt_pred = torch.optim.SGD(self.pred.parameters(), lr=self.learning_rate, weight_decay=1e-5, momentum=0.9)

        self.sample_per_class = torch.zeros(self.num_classes)
        trainloader = self.load_train_data()
        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()

                out_p = self.pred(rep.detach())
                loss = self.loss(out_g.detach() + out_p, y)
                self.opt_pred.zero_grad()
                loss.backward()
                self.opt_pred.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time



# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
