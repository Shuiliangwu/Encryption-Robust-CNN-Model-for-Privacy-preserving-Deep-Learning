import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
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


class ImageNet:
    def __init__(self, root, split):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            #                     0.229, 0.224, 0.225])
        ])
        self.dataset = datasets.ImageNet(
            root, split=split, transform=transform)
    def make_subset(self, start, end):
        indices_to_keep = [i for i, (_, class_idx) in enumerate(self.dataset.samples) if class_idx in range(start, end)]
        self.dataset = torch.utils.data.Subset(self.dataset, indices_to_keep)
        return self

    def loader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True, num_workers=0)
