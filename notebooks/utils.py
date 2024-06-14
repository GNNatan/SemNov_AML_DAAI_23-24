import matplotlib.pyplot as plt
import numpy as np
import torch
import warnings

lab_to_text_sr1 = {
    0: 'chair',
    1: 'shelf',
    2: 'door',
    3: 'sink',
    4: 'sofa',
    404: 'ood'
}

lab_to_text_sr2 = {
    0: 'bed',
    1: 'toilet',
    2: 'table',
    3: 'display',
    4: 'table',
    404: 'ood'
}

def load_from_file(filename):
    return torch.load(filename, map_location=torch.device('cpu'))

def render_cloud(cloud:np.ndarray,*, colors:np.array=None, cmap:str='RdYlGn', title:str = None, filename:str=None) -> None:
    x = cloud[:, 0]
    z = cloud[:, 1]
    y = cloud[:, 2]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.scatter(x, -y, z, c=colors, cmap=cmap)
    ax.axis('off')
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    plt.clf()
    plt.close()

