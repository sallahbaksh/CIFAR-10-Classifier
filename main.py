from MLP import MLP
import prepare_data


if __name__ == '__main__':
    trainX, trainY, testX, testY = prepare_data.prepare_data()
    mlp = MLP(trainX, trainY, testX, testY, eta=10**-3, epochs=250)
    mlp.create_architecture(f"Input, FullyConnected {trainX.shape[1]} 100,Sigmoid,"
                            f"FullyConnected 100 10, Sigmoid, DropOut 0.5, LogLoss")
    mlp.train()
    mlp.calculate_Accuracies()


