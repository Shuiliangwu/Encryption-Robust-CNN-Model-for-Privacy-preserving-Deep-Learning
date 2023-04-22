
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid


class AccuracyPlotter:
    def __init__(self, config, model_name):
        self.train_acc = np.loadtxt(f'./log/{model_name}/train_acc.txt')
        self.val_acc = np.loadtxt(f'./log/{model_name}/validation_acc.txt')
        self.epochs = range(1, config['num_epochs'] + 1)
        self.model_name = model_name

    def plot(self, round=1, show=True):
        plt.plot(self.epochs, self.train_acc, label='Training Accuracy')
        plt.plot(self.epochs, self.val_acc, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'./log/{self.model_name}/accuracy_{round}.png')
        if show:
            plt.show()


class Imshow:
    # Show the an array of images in a grid no need to unnormalize
    def imshow(img1, img2, path=None):
        if path is not None and os.path.exists('/'.join(path.split('/')[:-1])) == False:
            os.makedirs('/'.join(path.split('/')[:-1]))

        img = make_grid([img1, img2])
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        if path is not None:
            plt.savefig(path)
        plt.show()
