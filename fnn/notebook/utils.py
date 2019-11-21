import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_evals(names):
    saved_path = '/home/xhding/saved_model'
    fig, ax = plt.subplots()
    ax.set(xlabel='epoch', ylabel='correlation')
    for name in names:
        epochs = []
        pearsons = []
        eval_path = os.path.join(saved_path, name, 'eval.json')
        with open(eval_path, 'r') as f:
            history = json.load(f)
            for item in history:
                epochs.append(item['epoch'])
                pearsons.append(item['pearson'])
            ax.plot(epochs, pearsons, 'o-', label=name)
            ax.legend()
            
def plot_temperal_filters(path, layer):
    for checkpoint_path in os.scandir(path):
        if checkpoint_path.name.endswith("pth"):
            checkpoint = torch.load(os.path.join(path, checkpoint_path.name))
            filter_w = checkpoint['model_state_dict'][layer].cpu().numpy().squeeze()
            plt.plot(np.arange(filter_w.shape[0]), filter_w)
    plt.show()