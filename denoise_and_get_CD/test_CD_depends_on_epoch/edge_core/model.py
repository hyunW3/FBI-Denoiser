import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, dim,num_output_channel = 1, ngf_factor = 1,depth=4):
        """
            dim : number of channels in the first layer (colors : 3, grayscale : 1)
        """
        super(Generator, self).__init__()
        self.dim = dim
        self.image_to_features = []
        self.depth = depth
        self.first_1x1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * ngf_factor, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(self.dim*ngf_factor)
            )
        for i in range(depth):
            mul = 2**i * ngf_factor
            next_mul = 2**(i+1) * ngf_factor
            self.image_to_features.append(nn.Conv2d(mul * self.dim, next_mul * self.dim, kernel_size=4, stride=2,padding=1))
            # self.image_to_features.append(nn.LeakyReLU(0.2))
            self.image_to_features.append(nn.ReLU())
            self.image_to_features.append(nn.BatchNorm2d(next_mul * self.dim))
        self.image_to_features = torch.nn.ParameterList(self.image_to_features)

        self.features_to_image = []
        for i in range(depth):
            mul = 2**(depth-i) * ngf_factor
            next_mul = 2**(depth-1-i) * ngf_factor
            self.features_to_image.append(nn.ConvTranspose2d(mul * self.dim, next_mul * self.dim, kernel_size=4, stride=2, padding=1))
            # self.features_to_image.append(nn.LeakyReLU(0.2))
            self.features_to_image.append(nn.ReLU())
            self.features_to_image.append(nn.BatchNorm2d(next_mul* self.dim))
        self.features_to_image = torch.nn.ParameterList(self.features_to_image)
        self.last_1x1 = nn.Conv2d(next_mul*self.dim, num_output_channel, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Tanh()
    def forward(self, input_data):
        # UNet style generator
        features = []
        x = self.first_1x1(input_data)
        for layer in self.image_to_features:
            if isinstance(layer, nn.Conv2d):
                features.append(x)
            # print(layer,x.size())
            x = layer(x)
        # Return generated image
        for layer in self.features_to_image:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose2d) :
                # print(layer,x.size(),features[-1].size())
                x = x + features.pop()
        x = self.last_1x1(x)
        x = self.activation(x)
        return x