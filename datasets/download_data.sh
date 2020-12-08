echo "Downloading MNIST"
wget "http://deeplearning.net/data/mnist/mnist.pkl.gz"

echo "Downloading CIFAR-10"
wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
tar -xzvf cifar-10-python.tar.gz
mv cifar-10-batches-py cifar-10
