"""SegmentationNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
# from torchvision.models.segmentation.fcn import FCNHead


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        self.network = models.vgg16(pretrained = True).features[0:22] 
        self.conv = nn.Sequential (nn.Conv2d(512,256, kernel_size = 1),
                                    nn.ReLU (inplace = True),
                                    nn.Conv2d(256,num_classes, kernel_size = 1))


        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################


        x = self.network (x)
        x = self.conv (x)
        x = F.interpolate (x, size = (240,240), mode = "bilinear", align_corners = True )
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
