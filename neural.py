import numpy as np
import pandas as pd
import math
from collections import deque
from StringIO import StringIO


names = [i for i in range(0, 295)]
types = {i: np.int8 for i in range(3, 295)}
types.update({'0': np.float32, '1': np.float32, '2': np.unicode_})

nr_samples = 5000
dataset = pd.read_csv('datasets/tsmote0.csv', delimiter=',', nrows=nr_samples, names=names, dtype=types, header=None)

with open('datasets/tsmote0.csv', 'r') as f:
    q = deque(f, nr_samples)
dataset_bottom = pd.read_csv(StringIO(''.join(q)), header=None)
dataset = pd.concat([dataset, dataset_bottom])


Targets = []
for x in dataset.loc[:, 2].values:
    if x == "Chargeback":
        Targets.append(1)
    else:
        Targets.append(0)

dataset = dataset.drop(dataset.columns[2], axis=1)
Inputs = dataset.values
dataset = None
print("loaded")


# Training variables
epochs = 200
alpha = 0.1
samples = len(Inputs)

# Amount of neurons per layer
inputNeurons = len(Inputs[0])
hiddenNeurons = 100
outputNeurons = 1

# Weights for neurons
weightsInputHidden = np.random.rand(inputNeurons, hiddenNeurons)*2-1
weightsHiddenOutput = np.random.rand(hiddenNeurons)*2-1

# Used to store the errors in validation.
# meanErrors = np.zeros(1, epochs)
# amountErrors = 0


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1/(1 + math.exp(gamma))
    else:
        return 1/(1 + math.exp(-gamma))


# Sigmoid function on dot product of weights and inputs
def neuron(inputs, weights):
    threshold = 1
    result = sigmoid(np.dot(inputs, weights) - threshold)
    return result



# Training Neural network
for i in range(0, epochs):
    print("epoch: " + str(i))
    for j in range(0, samples):
        # calculate output hidden layer
        hidden_results = np.empty((hiddenNeurons))
        for k in range(0, hiddenNeurons):
            hidden_results[k] = neuron(Inputs[j], weightsInputHidden[:, k])
        # calculate output output layer
        output_results = neuron(hidden_results, weightsHiddenOutput)

        # update weights for output layer
        for k in range(0, outputNeurons):
            error = Targets[j] - output_results
            gradient = output_results * (1-output_results) * error
            for l in range(0, hiddenNeurons):
                dW = alpha * hidden_results[l] * gradient
                weightsHiddenOutput[k] = weightsHiddenOutput[k] + dW

        # update weights for hidden layer
        for k in range(0, hiddenNeurons):
            error = Targets[j] - output_results
            sum = weightsHiddenOutput[k] * output_results * (1-output_results) * error
            gradient = hidden_results[k] * (1-hidden_results[k]) * sum
            for l in range(0, inputNeurons):
                dW = alpha * Inputs[j, l] * gradient
                weightsInputHidden[l, k] = weightsInputHidden[l, k] + dW


weightsHiddenOutput.dump("temp/who0")
weightsInputHidden.dump("temp/wih0")

