import argparse
import io
from contextlib import redirect_stdout
import pickle

import numpy as np

import nn
from util import *
from layers import *
from trainer import Trainer

marks_map = {
                'XOR': [(90, 2), (80, 1)],
                'circle': [(90, 2), (80, 1)],
                'MNIST': [(90, 2), (80, 1)],
                'CIFAR10': [(35, 4), (32, 3), (30, 2), (25, 1)],
            }

def test_applications(dataset, seeds=[]):
    '''
    seeds -> list of seeds to test on
    '''
    print('-------------------------')
    print(f'Grading {dataset}')
    marks = 0

    if len(seeds)==0:
        np.random.seed(337)
        seeds = list(np.random.randint(1, 200, 10))

    acc = []
    for seed in seeds:
        trainer = Trainer(dataset)
        
        f = io.StringIO()
        with redirect_stdout(f):
            trainer.train(verbose=False)
        out = f.getvalue()
        acc.append(float(out.strip().split(' ')[-1]))

    if len(seeds)==10:
        acc = np.mean(np.sort(acc)[:4:-1])
    else:
        acc = np.mean(acc)

    for i, j in marks_map[dataset]:
        if acc > i:
            marks += j
            break

    print(f'Dataset: {dataset} Accuracy: {acc} Marks: {marks}')

    return marks

def test_cifar(seed=[]):
    '''
    seeds -> list of seeds to test on
    '''
    if len(seed) == 0:
        seed = 335
    else:
        seed = seed[0]

    print('-------------------------')
    print('Grading CIFAR10')
    marks = 0

    np.random.seed(seed)

    trainer = Trainer("CIFAR10")

    try:
        model = pickle.load(open('model.p', 'rb'))
    except:
        try:
           model = pickle.load(open('model.npy', 'rb'))
        except:
            print("Saved model not found")
            return 0 

    if 'ConvolutionLayer' not in [type(l).__name__ for l in trainer.nn.layers]:
        print('ConvolutionLayer not used')
        return 0

    try:
        i = 0
        for l in trainer.nn.layers:
            if type(l).__name__ not in ["AvgPoolingLayer", "MaxPoolingLayer", "FlattenLayer"]: 
                l.weights = model[i]
                l.biases = model[i+1]
                i = i + 2
        print("Model Loaded... ")
    except:
        print("Failed to load model")
        return 0

    _, acc = trainer.nn.validate(trainer.XTest, trainer.YTest)
    
    for i, j in marks_map['CIFAR10']:
        if acc > i:
            marks += j
            break

    print(f'Dataset: CIFAR10 Accuracy: {acc} Marks: {marks}')

    return marks

def grade_applications():
    marks = 0

    for d in ['XOR', 'circle', 'MNIST']:
        marks += test_applications(d)
    marks += test_cifar()

    return marks

def main(args):
    marks = 0
    if args.dataset is None:
        for d in ['XOR', 'circle', 'MNIST']:
            marks += test_applications(d, args.seeds)
        marks += test_cifar(args.seeds)
    elif args.dataset == 'CIFAR10':
        marks = test_cifar(args.seeds)
    else:
        marks = test_applications(args.dataset, args.seeds[:1])

    return marks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, choices=['MNIST', 'CIFAR10', 'XOR', 'circle'])
    parser.add_argument('--seeds', default=[], type=int, nargs='+')
    # Call as --dataset <DATASET> --seeds 1 2 3 4

    args = parser.parse_args()

    print('Total Marks', main(args))
