import os
import random
import numpy as np
from PIL import Image
import progressbar
import torchvision.transforms as transforms
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets


class Encryptor:
    # key_size: the width and height of the key matrix
    def __init__(self, key_size, keyFile=None):
        self.key_size = key_size
        if keyFile is None:
            self.generate_key()
        else:
            self.load_key(keyFile)

    # generate a random key matrix
    def generate_key(self):
        self.key = np.zeros((3, self.key_size**2, self.key_size**2))
        # initialize random seed from time
        random.seed(os.urandom(10))
        for k in range(3):
            for i in range(self.key_size**2):
                for j in range(self.key_size**2):
                    self.key[k][i][j] = random.random() * 2 - 1

    # save the key matrix to a file
    def save_key(self, keyFile='key.npy'):
        np.save(keyFile, self.key)

    # load the key matrix from a file
    def load_key(self, keyFile='key.npy'):
        self.key = np.load(keyFile)

    # encrypt the image with the key
    # image: the 3-channel image to be encrypted
    def encrypt(self, image):
        # get the width and height of the image
        width, height = 224, 224
        # get the image matrix by converting the tensor to numpy
        image_matrix = image.permute(1, 2, 0).numpy()
        # if the image is grayscale, convert it to RGB
        if len(image_matrix.shape) == 2:
            image_matrix = np.stack((image_matrix,)*3, axis=-1)

        # get the encrypted image matrix
        encrypted_matrix = np.zeros((width, height, 3))
        blocks = int(width / self.key_size)
        for i in range(blocks):
            for j in range(blocks):
                for k in range(3):  # for each channel
                    # get the block of the image
                    block = image_matrix[i*self.key_size:(
                        i+1)*self.key_size, j*self.key_size:(j+1)*self.key_size, k]
                    # get the encrypted block by flatten the block and multiply the key
                    encrypted_block = np.matmul(block.flatten(), self.key[k])
                    # reshape the encrypted block to the original size
                    encrypted_block = encrypted_block.reshape(
                        (self.key_size, self.key_size))
                    # put the encrypted block into the encrypted image matrix
                    encrypted_matrix[i*self.key_size:(i+1)*self.key_size, j*self.key_size:(
                        j+1)*self.key_size, k] = encrypted_block
        # return the encrypted image as float32 as npy format
        return encrypted_matrix.astype(np.float32)


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

    def make_subset(self, start_end):
        start, end = start_end
        indices_to_keep = [i for i, (_, class_idx) in enumerate(
            self.dataset.samples) if class_idx in range(start, end)]
        self.dataset = torch.utils.data.Subset(self.dataset, indices_to_keep)
        return self

    def loader(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=False, num_workers=0)


config = {
    'ImageNet_root': 'D:/ImageNet/',
    'num_classes': (601, 701),
}


class DatasetGenerator:
    def __init__(self, root, dataset_path, key_size):
        self.root = root
        self.key_size = key_size
        self.resolution = 224
        self.dataset_path = dataset_path
        self.encryptor = Encryptor(key_size)

    # get the image path list
    def get_image_list(self):
        self.image_list = []  # {ID: image path}
        for root, dirs, files in os.walk(self.root):
            for file in files:
                if file.endswith('.JPEG'):
                    ID = int(file.split('.')[0].split('_')[2])
                    path = os.path.join(root, file)
                    self.image_list.append((ID, path))

    # read labels from the file
    def read_labels(self, labelFile='ILSVRC2012_validation_ground_truth.txt'):
        self.labels = []
        with open(labelFile, 'r') as f:
            for line in f:
                self.labels.append(int(line))

    # generate the dataset
    def generate_dataset(self):
        testset = ImageNet(config['ImageNet_root'], 'val').make_subset(config['num_classes']).loader(
            batch_size=1)
        trainset = ImageNet(config['ImageNet_root'], 'train').make_subset(config['num_classes']).loader(
            batch_size=1)

        # generate the trainset
        if not os.path.exists(f'{self.dataset_path}/train'):
            os.makedirs(f'{self.dataset_path}/train')
        trainset_labels = open(f'{self.dataset_path}/train_label.txt', 'w+')

        print('Generating the trainset...')
        # initialize the progress bar
        widgets = ['Progress: ', progressbar.Percentage(), ' ',
                   progressbar.Bar(marker=progressbar.RotatingMarker()), ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(
            widgets=widgets, maxval=trainset.__len__())
        bar.start()
        for i, data in enumerate(trainset, 0):
            image, label = data
            image = image[0]
            label = label[0] - config['num_classes'][0]
            # encrypt and save the image to the trainset
            encrypted_image = self.encryptor.encrypt(image)
            if encrypted_image.shape != (self.resolution, self.resolution, 3):
                print(encrypted_image.shape)
            np.save(f'{self.dataset_path}/train/{i}.npy', encrypted_image)
            # write the label to the train_label.txt
            trainset_labels.write(f'{i}.npy {label}\n')
            # update the progress bar
            bar.update(i)

        bar.finish()
        # generate the testset
        if not os.path.exists(f'{self.dataset_path}/test'):
            os.makedirs(f'{self.dataset_path}/test')
        testset_labels = open(f'{self.dataset_path}/test_label.txt', 'w+')

        print('Generating the testset...')
        # initialize the progress bar
        widgets = ['Progress: ', progressbar.Percentage(), ' ',
                   progressbar.Bar(marker=progressbar.RotatingMarker()), ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(
            widgets=widgets, maxval=testset.__len__())
        bar.start()
        for i, data in enumerate(testset, 0):
            image, label = data
            image = image[0]
            label = label[0] - config['num_classes'][0]
            # encrypt and save the image to the testset
            encrypted_image = self.encryptor.encrypt(image)
            if encrypted_image.shape != (self.resolution, self.resolution, 3):
                print(encrypted_image.shape)
            np.save(f'{self.dataset_path}/test/{i}.npy', encrypted_image)
            # write the label to the test_label.txt
            testset_labels.write(f'{i}.npy {label}\n')
            # update the progress bar
            bar.update(i)
        bar.finish()
        print('Done!')


if __name__ == '__main__':
    DatasetGenerator = DatasetGenerator(
        root=config['ImageNet_root'], dataset_path='D:/dataset3/data3', key_size=16)
    DatasetGenerator.generate_dataset()
