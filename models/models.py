import torch
import torch.nn as nn
import torchvision.models as models


# Define the model: ResNet/ConvNeXt with a new convolutional layer and a new transposed convolutional layer


class Net(nn.Module):
    _models = {
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'convnext_tiny': models.convnext_tiny,
        'convnext_small': models.convnext_small,
        'convnext_base': models.convnext_base,
        'convnext_large': models.convnext_large,
    }

    def generate_adaptation_net(self, key_size, pretrained=False):
        self.conv_layer = nn.Conv2d(
            in_channels=3, out_channels=key_size**2*3, kernel_size=key_size, stride=key_size, padding=1)
        self.transposed_conv_layer = nn.ConvTranspose2d(
            in_channels=key_size**2*3, out_channels=3, kernel_size=key_size, stride=key_size, padding=0)
        
        if pretrained:
            state_dict = torch.load('./models/pretrained/best_model.pth')
            self.conv_layer.load_state_dict({'weight': state_dict['conv_layer.weight'], 'bias': state_dict['conv_layer.bias']})
            self.transposed_conv_layer.load_state_dict({'weight': state_dict['transposed_conv_layer.weight'], 'bias': state_dict['transposed_conv_layer.bias']})

        self.adaptation_net = nn.Sequential(
            self.conv_layer, self.transposed_conv_layer)

    def generate_cnn(self, model_name, num_classes, pretrained=False):
        model = self._models[model_name](pretrained=pretrained)

        # Replace the final fully connected layer with a new one that has num_classes output units
        if model_name.startswith('resnet'):
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

        elif model_name.startswith('convnext'):
            # Get the last block of the ConvNeXt model
            last_block = list(model.children())[-1]
            # Replace the final fully connected layer with a new one that has num_classes output units
            last_layer = nn.Linear(
                in_features=last_block[-1].in_features, out_features=num_classes, bias=True)
            last_block = nn.Sequential(
                last_block[0], last_block[1], last_layer)

            # Add modified last block to the model
            model = nn.Sequential(*list(model.children())[:-1], last_block)
        
        return model


    def __init__(self, key_size, num_classes, model_name, adaptation, adaptation_pretrained=False, cnn_pretrained=False):
        super(Net, self).__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        if self.model_name not in self._models:
            raise ValueError('Invalid model name: {}'.format(self.model_name))

        # Add the pretrained CNN
        self.model = self.generate_cnn(self.model_name, num_classes, pretrained = cnn_pretrained)

        # Add the adaptation net
        if adaptation:
            self.generate_adaptation_net(key_size, adaptation_pretrained)
            self.model = nn.Sequential(self.adaptation_net, self.model)

    def forward(self, x):
        return self.model(x)

    def forward_adaptation(self, x):
        if self.adaptation_net is None:
            raise ValueError('Adaptation net is not defined')
        return self.adaptation_net(x)
    
