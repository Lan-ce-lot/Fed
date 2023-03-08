
from threading import Thread
import time

from algorithms.client.clientFedbn import clientFedbn
from algorithms.server.server import Server


class FedBN(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(args, clientFedbn)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()


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
