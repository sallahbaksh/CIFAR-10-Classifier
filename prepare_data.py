import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def prepare_data():
    train_files, trainX_data, trainY_data = [], [], []
    prefix = "./cifar-10-batches-py/"

    [train_files.append(unpickle(prefix + f"data_batch_{i}")) for i in range(1, 6)]
    test_file = unpickle(prefix + "test_batch")

    [trainX_data.append(train_files[i][b'data']) for i in range(len(train_files))]
    trainX_data = np.array(np.concatenate(trainX_data))

    [trainY_data.append(train_files[i][b'labels']) for i in range(len(train_files))]
    trainY_data = np.array(np.concatenate(trainY_data))

    testX_data = np.array(test_file[b'data'])
    testY_data = np.array(test_file[b'labels'])

    return trainX_data, trainY_data, testX_data, testY_data
