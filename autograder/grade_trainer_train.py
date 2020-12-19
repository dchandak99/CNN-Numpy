import io
from contextlib import redirect_stdout

import numpy as np

import nn, nn_ref
from util import *
import layers_ref
import layers
from trainer import Trainer

def grade_trainer_train():

    print('-------------------------')
    print('Grading Trainer')

    marks = 0

    trainer = Trainer('XOR')

    np.random.seed(337)

    trainer.XTrain, trainer.YTrain, _, _, trainer.XTest, trainer.YTest = readXOR()
    trainer.batch_size = 10
    trainer.epochs = 15
    trainer.lr = 1e-3
    trainer.nn = nn_ref.NeuralNetwork(out_nodes=2, lr=trainer.lr)
    trainer.nn.addLayer(layers_ref.FullyConnectedLayer(2, 3, 'relu'))
    trainer.nn.addLayer(layers_ref.FullyConnectedLayer(3, 2, 'softmax'))

    f = io.StringIO()
    with redirect_stdout(f):
        trainer.train(verbose=False)

    out = f.getvalue()

    acc = float(out.strip().split(' ')[-1]) # 95.6

    if acc > 94.5:
        marks += 2
    elif acc > 90:
        marks += 1

    print('Marks for trainer', marks)

    return marks

if __name__ == '__main__':
    print(test_trainer_train())
