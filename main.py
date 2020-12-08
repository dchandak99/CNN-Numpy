'''Main script to run the code

Usage - python3 main.py --dataset [] --verbose --seed seed_value

'''

import argparse

import numpy as np

from trainer import * 
from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """ Arguments: arg """
    parser.add_argument('--dataset', choices=['MNIST', 'CIFAR10', 'XOR', 'circle'])
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--seed',type=int, default=335)

    
    args = parser.parse_args()

    np.random.seed(args.seed)
    trainer = Trainer(args.dataset)
    trainer.train(args.verbose)

