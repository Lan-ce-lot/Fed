import copy
from threading import Thread
import time

import numpy as np

from algorithms.client.clientFedbn import clientFedbn
from algorithms.server.server import Server
from dataset.digits import prepare_data


class FedBN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients_digits(args, clientFedbn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    def set_clients_digits(self, args, clientObj):
        train_loaders, test_loaders = prepare_data(args)

        # name of each client dataset
        datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']

        # federated setting
        client_num = len(datasets)
        client_weights = [1 / client_num for i in range(client_num)]
        for i in range(client_num):
            train_data_loader = train_loaders[i]
            test_data_loader = test_loaders[i]
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data_loader),
                               test_samples=len(test_data_loader),
                               train_loader=train_data_loader,
                               test_loader=test_data_loader
                              )
            self.clients.append(client)



    def train_bn(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train_bn()

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

    def train(self):
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))

        # self.save_results()
        # self.save_global_model()
