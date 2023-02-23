"""
MaskCycleGAN-VC models as described in https://arxiv.org/pdf/2102.12841.pdf
Inspired by https://github.com/jackaduma/CycleGAN-VC2
"""

import numpy as np

import torch
import torch.nn as nn
#from torchsummary import summary
import torch.nn.functional as F

class GLU(nn.Module):
    """Custom implementation of GLU since the paper assumes GLU won't reduce
    the dimension of tensor by 2.
    """

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class PixelShuffle(nn.Module):
    """Custom implementation pf Pixel Shuffle since PyTorch's PixelShuffle
    requires a 4D input (we have 3D inputs).
    """

    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        n = x.shape[0]
        c_out = x.shape[1] // 2
        w_new = x.shape[2] * 2
        return x.view(n, c_out, w_new)

class ResidualLayer(nn.Module):
    """ResBlock.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=1,
                                                    padding=padding),
                                          nn.InstanceNorm1d(num_features=out_channels,
                                                            affine=True))

        self.conv_layer_gates = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                        out_channels=out_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=out_channels,
                                                                affine=True))

        self.conv1d_out_layer = nn.Sequential(nn.Conv1d(in_channels=out_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        stride=1,
                                                        padding=padding),
                                              nn.InstanceNorm1d(num_features=in_channels,
                                                                affine=True))

    def forward(self, x):
        h1_norm = self.conv1d_layer(x)
        h1_gates_norm = self.conv_layer_gates(x)
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)  # GLU
        h2_norm = self.conv1d_out_layer(h1_glu)
        return x + h2_norm


class RN_B(nn.Module):
    def __init__(self, feature_channels):
        super(RN_B, self).__init__()
        '''
        input: tensor(features) x: (B,C,M,N)
               condition Mask: (B,1,H,W): 0 for background, 1 for foreground
        return: tensor RN_B(x): (N,C,M,N)
        ---------------------------------------
        args:
            feature_channels: C
        '''
        # RN
        #self.rn = RN_binarylabel(feature_channels)    # need no external parameters

        # gamma and beta
        self.foreground_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.foreground_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_gamma = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        self.background_beta = nn.Parameter(torch.zeros(feature_channels), requires_grad=True)
        #self.bn_norm=nn.BatchNorm1d(feature_channels, affine=False, track_running_stats=False)
        self.bn_norm=BN(affine=False)

    def forward(self, x, mask):
        # mask = F.adaptive_max_pool2d(mask, output_size=x.size()[2:])
        #mask = F.interpolate(mask, size=x.size()[2:], mode='nearest')   # after down-sampling, there can be all-zero mask
        
        mask1=torch.zeros(size=x.shape)
        #mask.to("cuda:0")
        #print(f"Mask shape {mask.shape}")
        #print(f"Image shape {x.shape}")
        for i in range(min(mask.shape[0],x.shape[0])):
          for j in range(min(x.shape[1],mask.shape[1])):
            for k in range(min(x.shape[2],mask.shape[2])):
              mask1[i][j][k]=mask[i][j][k];
        #rn_x = self.rn(x, mask)\
        #print(f"Mask shape {mask.shape}")
        mask1=mask1.to("cuda:0")
        rn_x_f,_,_=self.bn_norm(x*mask1)
        #print(f"RNf shape {rn_x_f.shape}")
        rn_x_b,_,_=self.bn_norm(x*(1-mask1))
        #print(f"RNb shape {rn_x_b.shape}")
        rn_x=rn_x_f+rn_x_b
        #print(f"rnx {rn_x.shape}")

        rn_x_foreground = (rn_x*mask1) * (1 + self.foreground_gamma[None,:,None]) + self.foreground_beta[None,:,None]
        rn_x_background = (rn_x*(1-mask1)) * (1 + self.background_gamma[None,:,None]) + self.background_beta[None,:,None]

        return rn_x_foreground + rn_x_background

class DownSampleGenerator(nn.Module):
    """Downsampling blocks of the Generator.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownSampleGenerator, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True))
                                       
                                                         
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding), nn.InstanceNorm2d(num_features=out_channels, affine=True))
                                                       
                                            

    def forward(self, x):
        # GLU
        return self.convLayer(x) * torch.sigmoid(self.convLayer_gates(x))





class Generator(nn.Module):
    """Generator of MaskCycleGAN-VC
    """

    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Generator, self).__init__()
        Cx, Tx = input_shape
        self.flattened_channels = (Cx // 4) * residual_in_channels
        #self.rn=RN_B(feature_channels=residual_in_channels)
        # 2D Conv Layer
        self.conv1 = nn.Conv2d(in_channels=2,
                               out_channels=residual_in_channels // 2,
                               kernel_size=(5, 15),
                               stride=(1, 1),
                               padding=(2, 7))

        self.conv1_gates = nn.Conv2d(in_channels=2,
                                     out_channels=residual_in_channels // 2,
                                     kernel_size=(5, 15),
                                     stride=1,
                                     padding=(2, 7))

        # 2D Downsampling Layers
        self.downSample1 = DownSampleGenerator(in_channels=residual_in_channels // 2,
                                               out_channels=residual_in_channels,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2)

        self.downSample2 = DownSampleGenerator(in_channels=residual_in_channels,
                                               out_channels=residual_in_channels,
                                               kernel_size=5,
                                               stride=2,
                                               padding=2)

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Conv1d(in_channels=self.flattened_channels,
                                         out_channels=residual_in_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.rn=RN_B(feature_channels=residual_in_channels)

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer2 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer3 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer4 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer5 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)
        self.residualLayer6 = ResidualLayer(in_channels=residual_in_channels,
                                            out_channels=residual_in_channels * 2,
                                            kernel_size=3,
                                            stride=1,
                                            padding=1)

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Conv1d(in_channels=residual_in_channels,
                                         out_channels=self.flattened_channels,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0)
        self.rn_1 = RN_B(feature_channels=self.flattened_channels)

        # UpSampling Layers
        self.upSample1 = self.upsample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 4,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        self.glu = GLU()

        self.upSample2 = self.upsample(in_channels=residual_in_channels,
                                       out_channels=residual_in_channels * 2,
                                       kernel_size=5,
                                       stride=1,
                                       padding=2)

        # 2D Conv Layer
        self.lastConvLayer = nn.Conv2d(in_channels=residual_in_channels // 2,
                                       out_channels=1,
                                       kernel_size=(5, 15),
                                       stride=(1, 1),
                                       padding=(2, 7))

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.ConvLayer = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm1d(
                                           num_features=out_channels,
                                           affine=True),
                                       GLU())

        return self.ConvLayer

    def upsample(self, in_channels, out_channels, kernel_size, stride, padding):
        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.PixelShuffle(upscale_factor=2),
                                       nn.InstanceNorm2d(
                                           num_features=out_channels // 4,
                                           affine=True),
                                       GLU())
        return self.convLayer

    def forward(self, x, mask):
        # Conv2d
        x = torch.stack((x*mask, mask), dim=1)
        #x=x*mask
        conv1 = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))  # GLU

        # Downsampling
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        # Reshape
        reshape2dto1d = downsample2.view(
            downsample2.size(0), self.flattened_channels, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)

        # 2D -> 1D
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        conv2dto1d_layer = self.rn(conv2dto1d_layer,mask)

        # Residual Blocks
        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        conv1dto2d_layer = self.rn_1(conv1dto2d_layer,mask)

        # Reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)

        # UpSampling
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        # Conv2d
        output = self.lastConvLayer(upsample_layer_2)
        output = output.squeeze(1)
        return output


class Discriminator(nn.Module):
    """PatchGAN discriminator.
    """

    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super(Discriminator, self).__init__()

        self.convLayer1 = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=residual_in_channels // 2,
                                                  kernel_size=(3, 3),
                                                  stride=(1, 1),
                                                  padding=(1, 1)),
                                        GLU())

        # Downsampling Layers
        self.downSample1 = self.downsample(in_channels=residual_in_channels // 2,
                                           out_channels=residual_in_channels,
                                           kernel_size=(3, 3),
                                           stride=(2, 2),
                                           padding=1)

        self.downSample2 = self.downsample(in_channels=residual_in_channels,
                                           out_channels=residual_in_channels * 2,
                                           kernel_size=(3, 3),
                                           stride=[2, 2],
                                           padding=1)

        self.downSample3 = self.downsample(in_channels=residual_in_channels * 2,
                                           out_channels=residual_in_channels * 4,
                                           kernel_size=[3, 3],
                                           stride=[2, 2],
                                           padding=1)

        self.downSample4 = self.downsample(in_channels=residual_in_channels * 4,
                                           out_channels=residual_in_channels * 4,
                                           kernel_size=[1, 10],
                                           stride=(1, 1),
                                           padding=(0, 2))

        # Conv Layer
        self.outputConvLayer = nn.Sequential(nn.Conv2d(in_channels=residual_in_channels * 4,
                                                       out_channels=1,
                                                       kernel_size=(1, 3),
                                                       stride=[1, 1],
                                                       padding=[0, 1]))

    def downsample(self, in_channels, out_channels, kernel_size, stride, padding):
        convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=padding),
                                  nn.InstanceNorm2d(num_features=out_channels,
                                                    affine=True),
                                  GLU())
        return convLayer

    def forward(self, x):
        # x has shape [batch_size, num_features, frames]
        # discriminator requires shape [batchSize, 1, num_features, frames]
        x = x.unsqueeze(1)
        conv_layer_1 = self.convLayer1(x)
        downsample1 = self.downSample1(conv_layer_1)
        downsample2 = self.downSample2(downsample1)
        downsample3 = self.downSample3(downsample2)
        fv=self.outputConvLayer(downsample3)
        output = torch.sigmoid(fv)
        #print(output.shape)
        return output, fv


#if __name__ == '__main__':
    # Non exhaustive test for MaskCycleGAN-VC models

    # Generator Dimensionality Testing
    #np.random.seed(0)

    residual_in_channels = 256
    # input = np.random.randn(2, 80, 64)
    #input = np.random.randn(2, 80, 64)
    #input = torch.from_numpy(input).float()
    #print("Generator input: ", input.shape)
    #mask = torch.ones_like(input)
    #mask.to("cuda:0")
    #generator = Generator(input.shape[1:], residual_in_channels)
    #output = generator(input, mask)
    #print("Generator output shape: ", output.shape)

    # Discriminator Dimensionality Testing
    #discriminator = Discriminator(input.shape[1:], residual_in_channels)
    #output = discriminator(output)
    #print("Discriminator output shape ", output.shape)
