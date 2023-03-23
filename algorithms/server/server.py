import torch
import os
import numpy as np

import copy
import time
import random

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from util.data_util import read_client_data


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.num_class = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        # self.train_slow_clients = []
        # self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        # self.train_slow_rate = args.train_slow_rate
        # self.send_slow_rate = args.send_slow_rate

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                              )
            self.clients.append(client)

    # random select slow clients
    # def select_slow_clients(self, slow_rate):
    #     slow_clients = [False for i in range(self.num_clients)]
    #     idx = [i for i in range(self.num_clients)]
    #     idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
    #     for i in idx_:
    #         slow_clients[i] = True
    #
    #     return slow_clients

    # def set_slow_clients(self):
    #     self.train_slow_clients = self.select_slow_clients(
    #         self.train_slow_rate)
    #     self.send_slow_clients = self.select_slow_clients(
    #         self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            join_clients = np.random.choice(range(self.join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            join_clients = self.join_clients
        selected_clients = list(np.random.choice(self.clients, join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                               client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)



    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_loss = []
        for c in self.clients:
            ct, ns, auc, loss = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)
            tot_loss.append(loss * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_loss
    def load_global_test_data(self, batch_size=None, dataset=None):
        if batch_size == None:
            batch_size = self.batch_size
        if dataset == 'Cifar':
            dataset = 'cifar100FL/cifar20'
        test_data_dir = os.path.join('./dataset', dataset, 'test/')
        test_file = test_data_dir + 'global' + '.npz'
        with open(test_file, 'rb') as f:
            data = np.load(f, allow_pickle=True)['data'].tolist()

        X_test = torch.Tensor(data['x']).type(torch.float32)
        y_test = torch.Tensor(data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=False)

    def test_generic_metric(self, num_classes, device, model, test_data=None):
        if test_data == None:
            test_loader_global = self.load_global_test_data(dataset=self.dataset)
        else:
            test_loader_global = test_data
        model.eval()

        test_acc = 0
        test_num = 0
        loss = 0
        y_prob = []
        y_true = []
        if isinstance(test_loader_global, list):
            for idx in range(len(test_loader_global)):
                with torch.no_grad():
                    for x, y in test_loader_global[idx]:
                        x = x.to(device)
                        y = y.to(device)
                        output = model(x)

                        test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                        test_num += y.shape[0]

                        y_prob.append(output.detach().cpu().numpy())
                        nc = num_classes
                        if num_classes == 2:
                            nc += 1
                        lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                        if num_classes == 2:
                            lb = lb[:, :2]
                        y_true.append(lb)

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        else:
            with torch.no_grad():
                for x, y in test_loader_global:
                    x = x.to(device)
                    y = y.to(device)
                    output = model(x)

                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_num += y.shape[0]

                    y_prob.append(output.detach().cpu().numpy())
                    nc = num_classes
                    if num_classes == 2:
                        nc += 1
                    lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                    if num_classes == 2:
                        lb = lb[:, :2]
                    y_true.append(lb)

            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)

            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')


        return test_acc, test_num, auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        test_loss = sum(stats[4]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Loss: {:.4f}".format(test_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))

        return train_loss, test_acc, test_loss

    def report_process(self, avg_acc, avg_train_loss, glo_acc, avg_test_loss, ):
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        import os

        TIMES = "{0:%Y-%m-%d--%H-%M-%S/}".format(datetime.now())
        ori_logs = str(self.dataset) + "-" + str(self.algorithm) + TIMES
        writer = SummaryWriter(log_dir=os.path.join('./log', ori_logs))
        for epoch in range(len(avg_acc)):
            writer.add_scalar("acc/avg_p_acc", avg_acc[epoch], epoch)
            writer.add_scalar("loss/avg_train_loss", avg_train_loss[epoch], epoch)
            writer.add_scalar("loss/avg_test_loss", avg_test_loss[epoch], epoch)
            writer.add_scalar("acc/g_acc", glo_acc[epoch], epoch)
        writer.close()

