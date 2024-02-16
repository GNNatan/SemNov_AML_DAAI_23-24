import matplotlib.pyplot as plt
import numpy as np
import torch

lab_to_text_sr1 = {
    0: 'chair',
    1: 'shelf',
    2: 'door',
    3: 'sink',
    4: 'sofa'
}

lab_to_text_sr2 = {
    0: 'bed',
    1: 'toilet',
    2: 'table',
    3: 'display',
    4: 'table'
}

def accuracy(samples, labels, truth, threshold) -> float:
    acc = 0  #sum of all correct predictions
    i = 0    #number of iterations
    pred = 0 #temporary variable to store the prediction relative to the current sample
    for s in samples:
        if s > threshold:
            pred = labels[i]
        else:
            pred = 404
        if pred == truth[i]:
            acc += 1
        i += 1
    return acc/i

def calculate_threshold(scores, preds, truth) -> float:
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    for t in thresholds:
        accuracies.append(accuracy(scores, preds, truth, t))

    plt.title('Threshold effect on model accuracy')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.plot(thresholds, accuracies)

    return thresholds[np.argmax(accuracies)]

def load_from_file(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

def render_cloud(cloud:np.ndarray, filename:str=None) -> None:
    x = cloud[:, 0]
    z = cloud[:, 1]
    y = cloud[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, -y, z)
    ax.axis('off')
    plt.show()

