import os
import pickle

import numpy as np
from tqdm import trange

from CNN import CNN, SGD, AdamGD
from Layers import dataloader, one_hot_encoding, resize_dataset, save_params_to_file, load_params_from_file, InputLayer
from ObjectiveFuncs import CrossEntropy


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def load_batch(filename):
    data = unpickle(filename)
    batch_x = [np.reshape(x, (3, 32, 32)) for x in data[b'data']]
    return batch_x, np.array(data[b'labels'])


def preprocess():
    train_x, train_y = [], []
    for batch in range(1, 6):
        batch_x, batch_y = load_batch(os.path.join('cifar-10-batches-py', f'data_batch_{batch}'))
        train_x.append(batch_x)
        train_y.append(batch_y)

    train_x = np.concatenate(train_x)
    train_y = np.concatenate(train_y)
    test_x, test_y = load_batch(os.path.join('cifar-10-batches-py', 'test_batch'))

    return train_x, train_y, test_x, test_y


def train(x_train, y_train, optimizer_choice):
    x_train = resize_dataset(x_train)
    x_train = InputLayer(x_train).forwardPropagate(x_train)
    y_train = one_hot_encoding(y_train)

    model = CNN()
    cost = CrossEntropy(None)
    eta = 0.005 if optimizer_choice != 1 else 0.0001

    optimizer = SGD(eta=eta, params=model.get_params()) if optimizer_choice != 1 else AdamGD(eta=eta, rho1=0.9,
                                                                                             rho2=0.999, epsilon=1e-8,
                                                                                             params=model.get_params())
    train_costs = []

    print("----------------TRAINING-----------------\n")

    num_epochs = 20
    batch_size = 1

    num_train_obs = len(x_train)

    loading_bar = None

    for epoch in range(num_epochs):
        loading_bar = trange(num_train_obs // batch_size)

        train_loss = 0
        train_acc = 0
        train_loader = dataloader(x_train, y_train, batch_size)

        for i, (X_batch, y_batch) in zip(loading_bar, train_loader):
            y_pred = model.forwardPropagate(X_batch)
            cost.y = y_batch
            loss = cost.eval(y_pred)

            grads = model.backwardPropagate(y_pred, y_batch)
            params = optimizer.update_params(grads)
            model.set_params(params)

            train_loss += loss * batch_size
            train_acc += sum(np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1))

            loading_bar.set_description("[Train] Epoch {}".format(epoch + 1))

        train_loss /= num_train_obs
        train_costs.append(train_loss)
        train_acc /= num_train_obs

        info_train = "train-loss: {:0.6f} | train-acc: {:0.3f}"
        print(info_train.format(train_loss, train_acc))

        save_params_to_file(model, optimizer_choice)
        print()

    loading_bar.close()


def eval_efficiency(x_train, x_test, y_test, optimizer_choice):
    x_test = InputLayer(x_train).forwardPropagate(x_test)
    x_test = resize_dataset(x_test)
    y_test = one_hot_encoding(y_test)

    cost = CrossEntropy(None)
    model = CNN()
    model = load_params_from_file(model, optimizer_choice)

    print("--------------------EVALUATION-------------------\n")

    batch_size = 1

    num_test_obs = len(x_test)
    test_loss = 0
    test_accuracy = 0

    loading_bar = trange(num_test_obs // batch_size)
    test_loader = dataloader(x_test, y_test, batch_size)

    for i, (x_batch, y_batch) in zip(loading_bar, test_loader):
        y_pred = model.forwardPropagate(x_batch)
        cost.y = y_batch
        loss = cost.eval(y_pred)

        test_loss += loss * batch_size
        test_accuracy += sum(np.argmax(y_batch, axis=1) == np.argmax(y_pred, axis=1))

        loading_bar.set_description("Evaluation")

    test_loss /= num_test_obs
    test_accuracy /= num_test_obs

    print("test-loss: {:0.6f} | test-acc: {:0.3f}".format(test_loss, test_accuracy))


if __name__ == '__main__':
    choice = int(input("1 for ADAM, 0 for Reg:\n"))
    train_x, train_y, test_x, test_y = preprocess()
    train(train_x, train_y, choice)
    eval_efficiency(train_x, test_x, test_y, choice)
