import os
import random
import numpy as np
from PIL import Image
import progressbar


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
        width, height = image.size
        # get the image matrix
        image_matrix = np.array(image)
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
                    encrypted_block = np.dot(block.flatten(), self.key[k])
                    # reshape the encrypted block to the original size
                    encrypted_block = encrypted_block.reshape(
                        (self.key_size, self.key_size))
                    # put the encrypted block into the encrypted image matrix
                    encrypted_matrix[i*self.key_size:(i+1)*self.key_size, j*self.key_size:(
                        j+1)*self.key_size, k] = encrypted_block
        # return the encrypted image as float32 as npy format
        return encrypted_matrix.astype(np.float32)


class DatasetGenerator:
    def __init__(self, root, resolution, dataset_path, key_size, trainset_ratio=0.8):
        self.root = root
        self.key_size = key_size
        self.dataset_path = dataset_path
        self.trainset_ratio = trainset_ratio
        self.resolution = resolution
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
        self.get_image_list()
        self.read_labels()
        # shuffle the image list
        random.shuffle(self.image_list)
        # split the image list into trainset and testset
        self.trainset_image_list = self.image_list[:int(
            len(self.image_list) * self.trainset_ratio)]
        self.testset_image_list = self.image_list[int(
            len(self.image_list) * self.trainset_ratio):]

        # generate the trainset
        if not os.path.exists(f'{self.dataset_path}/train'):
            os.makedirs(f'{self.dataset_path}/train')
        trainset_labels = open(f'{self.dataset_path}/train_label.txt', 'w+')

        print('Generating the trainset...')
        # initialize the progress bar
        widgets = ['Progress: ', progressbar.Percentage(), ' ',
                   progressbar.Bar(marker=progressbar.RotatingMarker()), ' ', progressbar.ETA()]
        bar = progressbar.ProgressBar(
            widgets=widgets, maxval=len(self.trainset_image_list))
        bar.start()
        for i in range(len(self.trainset_image_list)):
            ID, image_path = self.trainset_image_list[i]
            image = Image.open(image_path)
            # resize the image to the resolution
            image = image.resize((self.resolution, self.resolution))
            # encrypt and save the image to the trainset
            encrypted_image = self.encryptor.encrypt(image)
            np.save(f'{self.dataset_path}/train/{i}.npy', encrypted_image)
            # write the label to the train_label.txt
            trainset_labels.write(f'{i}.npy {self.labels[ID-1]}\n')
            # update the progress bar
            bar.update(i)

        bar.finish()
        # generate the testset
        if not os.path.exists(f'{self.dataset_path}/test'):
            os.makedirs(f'{self.dataset_path}/test')
        testset_labels = open(f'{self.dataset_path}/test_label.txt', 'w+')

        print('Generating the testset...')
        # initialize the progress bar
        bar.start()

        for i in range(len(self.testset_image_list)):
            ID, image_path = self.testset_image_list[i]
            image = Image.open(image_path)
            # resize the image to the resolution
            image = image.resize((self.resolution, self.resolution))

            # encrypt and save the image to the testset
            encrypted_image = self.encryptor.encrypt(image)
            np.save(f'{self.dataset_path}/test/{i}.npy', encrypted_image)
            # write the label to the test_label.txt
            testset_labels.write(f'{i}.npy {self.labels[ID-1]}\n')
            # update the progress bar
            bar.update(i)

        bar.finish()

        print('Done!')


if __name__ == '__main__':
    DatasetGenerator = DatasetGenerator(
        root='ILSVRC2012_img_val', resolution=224, dataset_path='./dataset2/data2', key_size=8, trainset_ratio=0.85)
    DatasetGenerator.generate_dataset()
