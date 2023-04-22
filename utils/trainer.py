from visualize.visualizer import Imshow
from utils.config_loader import ConfigLoader
from datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import sys
import os
import shutil

sys.path.append('..')


class Trainer:

    def __init__(self, config, trainset_loader, testset_loader, net, device):
        self.config = config
        self.trainset_loader = trainset_loader
        self.testset_loader = testset_loader
        self.net = net.to(device)
        self.device = device

    def train(self, round=1, resume=False):
        # Define the loss function and the optimizer
        criterion = nn.CrossEntropyLoss()
        if self.config['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.net.parameters(
            ), lr=self.config['learning_rate'], momentum=0.9)
        elif self.config['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.net.parameters(),
                                   lr=self.config['learning_rate'])
        else:
            raise Exception('Optimizer not supported')

        epochNum = self.config['num_epochs']
        model_name = ConfigLoader.generate_model_name(self.config)

        # Load the optimizer state if resume is True and the optimizer file exists
        if resume and os.path.exists(f'./models/pretrained/optimizer.pth'):
            optimizer.load_state_dict(torch.load(
                f'./models/pretrained/optimizer.pth'))
            print('Optimizer loaded')

        # Creat the directory to store the log file if it does not exist
        if not os.path.exists(f'./log/{model_name}'):
            os.makedirs(f'./log/{model_name}')

        # Create a log file
        f = open(f'./log/{model_name}/train.log', 'a+')

        if round == 1:
            # Write configuration to the log file if it is the first round
            f.write(
                f'**************** CONFIGURATION ****************\r {self.config}\r')
            # Write timestamp to the log file if it is the first round
            f.write(
                f'**************** TIMESTAMP ****************\r {datetime.now()}\r')

        # Write the round number to the log file
        print(f'**************** ROUND {round} ****************')
        f.write(f'**************** ROUND {round} ****************\r')

        # Use List to store the loss and accuracy
        train_acc_list = []
        test_acc_list = []

        # Train the model
        best_acc = 0.0
        for epoch in range(epochNum):  # loop over the dataset multiple times
            # Training code
            running_loss = 0.0
            correct = 0
            total = 0

            # 0 is the starting index, i is the index of the batch
            for i, data in enumerate(self.trainset_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # load inputs and labels into GPU
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # print statistics
                running_loss += loss.item()
                train_acc = 100 * correct / total
                # print every 1% of the training set
                if i % (len(self.trainset_loader) // 100) == (len(self.trainset_loader) // 100) - 1:
                    print(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f} accuracy: {train_acc:.2f}%')
                    f.write(
                        f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f} accuracy: {train_acc:.2f}%\n')
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    # Show and save the image after the data augmentation
                    if self.config['view_adaptated_image']:
                        Imshow.imshow(inputs[0], self.net.forward_adaptation(
                            inputs[0]), f'./log/{model_name}/adapted_images/epoch{epoch + 1}_batch{i + 1}.png')

            # saving the training accuracy and its corresponding epoch
            print(f'Epoch {epoch + 1}: training accuracy = {train_acc:.2f}%')
            f.write(
                f'Epoch {epoch + 1}: training accuracy = {train_acc:.2f}%\n')
            train_acc_list.append(train_acc)

            print('********** VALIDATION **********')
            f.write('********** VALIDATION **********\n')

            # Validation code
            # set the network to evaluation mode
            self.net.eval()

            # evaluate the model on the test set

            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data in self.testset_loader:
                    images, labels = data

                    # load inputs and labels into GPU
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            print(f'Epoch {epoch + 1}: validation accuracy = {val_acc:.2f}%')
            f.write(
                f'Epoch {epoch + 1}: validation accuracy = {val_acc:.2f}%\n')
            test_acc_list.append(val_acc)

            # Save the model with the best validation accuracy
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.net.state_dict(),
                           f'./log/{model_name}/best_model.pth')

        # torch.save(net.state_dict(), 'model.pth')  to delete
        print('********** Finished Training **********')

        # Copy the model to the ./model/pretrained folder
        shutil.copy(f'./log/{model_name}/best_model.pth',
                    f'./models/pretrained/best_model.pth')

        # Save state dict of optimizer
        torch.save(optimizer.state_dict(), f'./log/{model_name}/optimizer.pth')
        torch.save(optimizer.state_dict(),
                   f'./models/pretrained/optimizer.pth')

        # Close the log file
        f.close()

        # Save the training and validation accuracy to txt
        np.savetxt(f'./log/{model_name}/train_acc.txt', train_acc_list)
        np.savetxt(f'./log/{model_name}/validation_acc.txt', test_acc_list)

        return best_acc

    def test(self):
        # Load the best model
        model_name = self.generate_model_name()
        self.net.load_state_dict(torch.load(
            f'./log/{model_name}/best_model.pth'))

        # set the network to evaluation mode
        self.net.eval()

        # evaluate the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testset_loader:
                images, labels = data

                # load inputs and labels into GPU
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f'Test accuracy = {acc:.2f}%')

        return acc
