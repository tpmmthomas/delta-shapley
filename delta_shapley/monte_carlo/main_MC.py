import argparse
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from delta_shapley.data import *
from delta_shapley.models import *
from delta_shapley.train import train
from delta_shapley.val import evaluate_model
from monte_carlo.MonteCarloShapley import MonteCarloShapley
import torch
import time
import os
import pandas as pd


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def get_parser():
    """
    Generate a parameters parser.
    """
    parser = argparse.ArgumentParser(description='Privacy attack parameters')

    # overall experiment parameters
    parser.add_argument("--save_dir", type=str, default='results')

     # data and model parameters
    parser.add_argument("--dataset", type=str, default='fashionmnist')
    parser.add_argument("--model", type=str, default='cnn')
    parser.add_argument("--pretrain", type=bool_flag, default=False)
    parser.add_argument("--datasize", type=int, default=100)


    # training parameters
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    # give index of datapoint to be evaluated
    parser.add_argument("--datapoint", type = int, default = 0) #seems useless
    parser.add_argument("--eval_datapoint_ids", type = list, default = [i for i in range(1)])


    return parser


if __name__ == '__main__':

    parser = get_parser()
    params = parser.parse_args()
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('running experiment', flush=True)
    if not os.path.exists(params.save_dir):
        os.makedirs(params.save_dir)
        print("New directory created!")

    trainset, testset, num_classes = load_data(params, binary_trainsize=params.datasize, binary_testsize=params.datasize)

    print("loaded trainset")
    MC = MonteCarloShapley(trainset, testset, L = 1,  beta = 1, c = 1, a = 0.05, b = 0.05, sup = 5, num_classes = num_classes, params = params, train_func = train, val_func = evaluate_model)
    print("starting run")
    start = time.time()
    shapleyvalues = MC.run(params.eval_datapoint_ids, params)
    end = time.time()

    print("Shapley Value of datapoints: " + str(params.eval_datapoint_ids)+" is "+ str(shapleyvalues))
    print("Total time taken: " + str(end-start) + " seconds")
