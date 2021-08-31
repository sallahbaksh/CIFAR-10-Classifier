import numpy as np
import pandas as pd
from MLP import MLP
from pprint import pprint


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


if __name__ == '__main__':
    data_directory = "./cifar-10-python/cifar-10-batches-py"
    label_names = unpickle(f"{data_directory}/batches.meta")
    train = []
    for i in range(1, 6):
        train.append(unpickle(f"{data_directory}/data_batch_{i}"))
    test = unpickle(f"{data_directory}/test_batch")
    # train = np.concatenate(train)
    try:
        # mlp = MLP(tx := train[b"data"], np.array(train[b"labels"]), test[b"data"], np.array(test[b"labels"]))
        train_x = pd.read_csv('../mnist_train_100.csv', header=None)
        test_x = pd.read_csv('../mnist_valid_10.csv', header=None)
        train_y = train_x.pop(0)
        test_y = test_x.pop(0)
        test_x = test_x.to_numpy()
        train_x = train_x.to_numpy()
        mlp = MLP(tx := train_x, train_y, test_x, test_y)
        mlp.create_architecture(f"Input, FullyConnected {tx.shape[1]} 10, Sigmoid, LogLoss")
        mlp.train()
        mlp.calculate_Accuracies()
    except Exception as e:
        print(e)

