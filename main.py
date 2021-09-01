from MLP import MLP
import prepare_data


if __name__ == '__main__':
    trainX, trainY, testX, testY = prepare_data.prepare_data()
    mlp = MLP(trainX, trainY, testX, testY, eta=10**-4, epochs=250)
    mlp.create_architecture(f"Input, FullyConnected {trainX.shape[1]} 10, Sigmoid, LogLoss")
    mlp.train()
    mlp.calculate_Accuracies()


