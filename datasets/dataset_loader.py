import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import sys
sys.path.append('..')


class MyDataSet(Dataset):

    # Define the transforms

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    def __init__(self, img_dir, label_dir):
        self.root = img_dir
        self.transform = self.transform
        self.img_dir = img_dir
        with open(label_dir, 'r') as f:
            self.labels = f.readlines()
            for i in range(len(self.labels)):
                self.labels[i] = int(self.labels[i].strip().split(' ')[1])

    def __getitem__(self, index):
        image = np.load(self.img_dir + str(index) + '.npy')
        if self.transform:
            # Apply the transforms, returns a torch tensor
            image = self.transform(image)
            image = image.float()
        else:
            image = torch.from_numpy(image)  # Convert to torch tensor
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)
