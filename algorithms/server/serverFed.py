import time

from threading import Thread

from algorithms.client.clientFed import clientFedours
from algorithms.client.clientFedavg import clientFedavg
from algorithms.server.server import Server
from dataset.cifar100 import prepare_data_cifar100
from dataset.digits import prepare_data
from util.utils import generate_projection_matrix


class FedOurs(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        QR = False
        self.proj_matrices = generate_projection_matrix(args.num_clients + 1, feature_dim=args.model.head.in_features,
                                                        qr=QR)

        self.set_clients_cifar100(args, clientObj=clientFedours)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

    def set_clients_cifar100(self, args, clientObj):
        train_loaders, test_loaders = prepare_data_cifar100(args)
        for i in range(args.num_clients):
            train_data_loader = train_loaders[i]
            test_data_loader = test_loaders[i]
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data_loader),
                               test_samples=len(test_data_loader),
                               train_loader=train_data_loader,
                               test_loader=test_data_loader,
                               g_proj=self.proj_matrices[0],
                               p_proj=self.proj_matrices[i+1],
                              )
            self.clients.append(client)

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
                               test_loader=test_data_loader,
                               data_name=datasets[i],
                               g_proj=self.proj_matrices[0],
                               p_proj=self.proj_matrices[i+1],
                              )
            self.clients.append(client)

    # def train_bn(self):
    #     for i in range(self.global_rounds+1):
    #         self.selected_clients = self.select_clients()
    #         self.send_models()
    #
    #         if i % self.eval_gap == 0:
    #             print(f"\n-------------Round number: {i}-------------")
    #             print("\nEvaluate global model")
    #             self.evaluate()
    #
    #         for client in self.selected_clients:
    #             client.train_bn()
    #
    #         self.receive_models()
    #         self.aggregate_parameters()
    #
    #     print("\nBest global accuracy.")
    #     print(max(self.rs_test_acc))
    #
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
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

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # self.save_results()
        # self.save_global_model()
