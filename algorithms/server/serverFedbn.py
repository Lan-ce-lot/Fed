import copy
from threading import Thread
import time

import numpy as np

from algorithms.client.clientFedbn import clientFedbn
from algorithms.server.server import Server
from dataset.cifar100 import prepare_data_cifar100
from dataset.digits import prepare_data_digits
from dataset.office_caltech_10 import prepare_data_office


class FedBN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        if args.dataset == 'digits' or args.dataset == 'office':
            self.set_clients_bn(args, clientObj=clientFedbn)
        else:
            self.set_clients(args, clientObj=clientFedbn)
        # elif args.dataset == 'Cifar':
        #     self.set_clients_cifar100(args, clientObj=clientFedbn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()

    # def set_clients_cifar100(self, args, clientObj):
    #     train_loaders, test_loaders = prepare_data_cifar100(args)
    #     for i in range(args.num_clients):
    #         train_data_loader = train_loaders[i]
    #         test_data_loader = test_loaders[i]
    #         client = clientObj(args,
    #                            id=i,
    #                            train_samples=len(train_data_loader),
    #                            test_samples=len(test_data_loader),
    #                            train_loader=train_data_loader,
    #                            test_loader=test_data_loader,
    #                           )
    #         self.clients.append(client)

    def set_clients_bn(self, args, clientObj):
        if args.dataset == "office":
            train_loaders, test_loaders = prepare_data_office(args)
            # name of each dataset
            datasets = ['Amazon', 'Caltech', 'DSLR', 'Webcam']
        elif args.dataset == "digits":
            train_loaders, test_loaders = prepare_data_digits(args)
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
                               test_loader=test_data_loader,
                               data_name=datasets[i]
                              )
            self.clients.append(client)



    def train(self):
        avg_acc, avg_train_loss, glo_acc = [], [], []
        for i in range(self.global_rounds+1):
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                test_acc, test_num, auc = self.test_generic_metric(self.num_class, self.device, self.global_model)
                print("Global Test Accurancy: {:.4f}".format(test_acc / test_num))
                print("Global Test AUC: {:.4f}".format(auc))
                avg_acc.append(test_acc / test_num)

                train_loss, avg_test_acc = self.evaluate()
                avg_train_loss.append(train_loss)
                glo_acc.append(avg_test_acc)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        self.report_process(avg_acc, avg_train_loss, glo_acc)

    def train_label_skew(self):
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
