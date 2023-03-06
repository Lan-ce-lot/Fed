import torch
import argparse
import os
import time
import warnings
import numpy as np
import torchvision
import logging

from algorithms.server.serverFedavg import FedAvg
from models.LeNet import LeNet
from options import args_parser

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

# hyper-params for Text tasks
vocab_size = 98635
max_len = 200
hidden_dim = 32


def run(args):
    time_list = []

    # reporter = MemReporter()
    model_str = args.model


    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "cnn":
            if args.dataset[:5] == "mnist" or args.dataset == "fmnist":
                args.model = LeNet().to(args.device)
        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            server = FedAvg(args, i)


        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    # Global average
    # average_data(dataset=args.dataset,
    #              algorithm=args.algorithm,
    #              goal=args.goal,
    #              times=args.times,
    #              length=args.global_rounds / args.eval_gap + 1)

    print("All done!")

    # reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    args = args_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_steps))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Time select: {}".format(args.time_select))
    print("Time threthold: {}".format(args.time_threthold))
    print("Global rounds: {}".format(args.global_rounds))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Local model: {}".format(args.model))
    print("Using device: {}".format(args.device))

    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("=" * 50)


    run(args)

