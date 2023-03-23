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
        self.id = id
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5, momentum=0.9)

        self.p_fea = copy.deepcopy(self.model.g_fea)
        self.opt_pfea = torch.optim.SGD(self.p_fea.parameters(), lr=self.learning_rate, weight_decay=1e-5, momentum=0.9)

        if self.dataset == "digits" or self.dataset == "office":
            # self.data_name = kwargs['data_name']
            self.train_loader = kwargs['train_loader']
            self.test_loader = kwargs['test_loader']
        self.g_proj = kwargs['g_proj']
        self.p_proj = kwargs['p_proj']

    def train(self):
        if self.dataset == "digits" or self.dataset == "office":
            trainloader = self.train_loader
        else:
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
                g_proj = torch.Tensor(self.g_proj).to(self.device)
                p_proj = torch.Tensor(self.p_proj).to(self.device)
                share_fea = self.model.base(x)
                g_fea = self.model.g_fea(share_fea)
                p_fea = self.p_fea(share_fea)
                g_fea = torch.mm(g_fea, g_proj)
                p_fea = torch.mm(p_fea, p_proj)
                fea = g_fea + p_fea
                # fea = torch.cat([g_fea, p_fea], dim=1)
                self.optimizer.zero_grad()
                self.opt_pfea.zero_grad()
                output = self.model.head(fea)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
                self.opt_pfea.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self):
        if self.dataset == "digits" or self.dataset == "office":
            train_loader = self.train_loader
        else:
            train_loader = self.load_train_data()
        self.model.eval()
        train_acc = 0
        train_num = 0
        loss = 0
        for x, y in train_loader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            g_proj = torch.Tensor(self.g_proj).to(self.device)
            p_proj = torch.Tensor(self.p_proj).to(self.device)
            share_fea = self.model.base(x)
            g_fea = self.model.g_fea(share_fea)
            p_fea = self.p_fea(share_fea)
            g_fea = torch.mm(g_fea, g_proj)
            p_fea = torch.mm(p_fea, p_proj)
            fea = g_fea + p_fea
            # fea = torch.cat([g_fea, p_fea], dim=1)
            output = self.model.head(fea)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            train_num += y.shape[0]
            loss += self.loss(output, y).item() * y.shape[0]

        # print(self.id, "_trainLoss:", loss * 1.0 / train_num, end=' ')
        # print(self.id, "_trainAcc:", train_acc * 1.0 / train_num)

        return loss, train_num
    def test_metrics(self):
        if self.dataset == "digits" or self.dataset == "office":
            test_loader_full = self.test_loader
        else:
            test_loader_full = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        loss = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in test_loader_full:
                x = x.to(self.device)
                y = y.to(self.device)
                g_proj = torch.Tensor(self.g_proj).to(self.device)
                p_proj = torch.Tensor(self.p_proj).to(self.device)
                share_fea = self.model.base(x)
                g_fea = self.model.g_fea(share_fea)
                p_fea = self.p_fea(share_fea)
                g_fea = torch.mm(g_fea, g_proj)
                p_fea = torch.mm(p_fea, p_proj)
                fea = g_fea + p_fea
                # fea = torch.cat([g_fea, p_fea], dim=1)
                output = self.model.head(fea)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                loss += self.loss(output, y).item() * y.shape[0]
                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)


        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        # print(self.id, "_testAcc:", test_acc * 1.0 / test_num)
        return test_acc, test_num, auc, loss

