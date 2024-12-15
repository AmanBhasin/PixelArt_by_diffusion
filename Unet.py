import torch
import torch.nn as nn
import numpy as np

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),   
            nn.BatchNorm2d(out_channels),  
            nn.ReLU(),   
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        if self.is_res:
            if self.same_channels:
                out = input + out2
            else:
                res_output = nn.Conv2d(input.shape[1], out2.shape[1], kernel_size=1, stride=1, padding=0).to(input.device)
                out = res_output(input) + out2
            
            return out / np.sqrt(2)
        else:
            return out2


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),  # Upsample the input tensor
            DoubleConvBlock(out_channels, out_channels),
            DoubleConvBlock(out_channels, out_channels),
        )

    def forward(self, input, skip):
        # Concatenate the input tensor input with the skip connection tensor along the channel dimension
        input = torch.cat((input, skip), 1)
        input = self.layers(input)
        return input


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleBlock, self).__init__()

        self.layers = nn.Sequential(
            DoubleConvBlock(in_channels, out_channels),
            DoubleConvBlock(out_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.layers(x)


class MLPBlock(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(MLPBlock, self).__init__()
        self.input_dim = input_dim
        
        # simple MLP with 2 hidden layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten the input
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_feature_maps=256, context_size=10, image_size=28) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_feature_maps = num_feature_maps
        self.context_size = context_size
        self.image_size = image_size

        self.convLayer = DoubleConvBlock(in_channels, num_feature_maps, is_res=True)

        self.down1 = DownSampleBlock(num_feature_maps, num_feature_maps)
        self.down2 = DownSampleBlock(num_feature_maps, 2*num_feature_maps)

        self.map_to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.time_embedding = MLPBlock(1, 2*num_feature_maps)
        self.time_embedding2 = MLPBlock(1, num_feature_maps)
        self.context_embedding = MLPBlock(context_size, 2*num_feature_maps)
        self.context_embedding2 = MLPBlock(context_size, num_feature_maps)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*num_feature_maps, 2*num_feature_maps, self.image_size//4, self.image_size//4),
            nn.GroupNorm(8, 2*num_feature_maps),
            nn.ReLU(),
        )

        self.up1 = UpsampleBlock(4*num_feature_maps, num_feature_maps)
        self.up2 = UpsampleBlock(2*num_feature_maps, num_feature_maps)

        self.final = nn.Conv2d(num_feature_maps, in_channels, 1)

    def forward(self, x, t, c=None):
        x = self.convLayer(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)

        latentvec = self.map_to_vec(down2)

        if c is None:
            c = torch.zeros(x.shape[0], self.context_size).to(x)

        cemb1 = self.context_embedding(c).view(-1, self.num_feature_maps * 2, 1, 1)
        temb1 = self.time_embedding(t).view(-1, self.num_feature_maps * 2, 1, 1)
        cemb2 = self.context_embedding2(c).view(-1, self.num_feature_maps, 1, 1)
        temb2 = self.time_embedding2(t).view(-1, self.num_feature_maps, 1, 1)

        up1 = self.up0(latentvec)
        up2 = self.up1(cemb1*up1 + temb1, down2)
        up3 = self.up2(cemb2*up2 + temb2, down1)

        out = self.final(up3)
        return out
