import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        self.conv_2d1 = nn.Conv2d (1, 32, kernel_size = 4, stride = 1, padding = 0)
        self.maxpool_2d1 = nn.MaxPool2d (2)
        self.dropout_1 = nn.Dropout (p = 0.1)
        
        self.conv_2d2 = nn.Conv2d (32, 64, kernel_size = 3, stride = 1, padding = 0)
        self.maxpool_2d2 = nn.MaxPool2d (2)
        self.dropout_2 = nn.Dropout (p = 0.2)
        
        self.conv_2d3 = nn.Conv2d (64, 128, kernel_size = 2, stride = 1, padding = 0)
        self.maxpool_2d3 = nn.MaxPool2d (2)
        self.dropout_3 = nn.Dropout (0.3)
        
        self.conv_2d4 = nn.Conv2d (128, 256, kernel_size = 1, stride = 1, padding = 0)
        self.maxpool_2d4 = nn.MaxPool2d (2)
        self.dropout_4 = nn.Dropout (0.4)

        self.fc1 = nn.Linear(6400, 1000)
        self.dropout_5 = nn.Dropout (0.5)

        self.fc2 = nn.Linear(1000, 1000)
        self.dropout_6 = nn.Dropout (0.6)

        self.fc3 = nn.Linear(1000, 30)

        self.convnet1 = nn.Sequential (
                    self.conv_2d1,
                    self.elu,
                    self.maxpool_2d1 ,
                    self.dropout_1 )
        
        self.convnet2 = nn.Sequential (               
                    self.conv_2d2 ,
                    self.elu,
                    self.maxpool_2d2 ,
                    self.dropout_2 )
        
        self.convnet3 = nn.Sequential (
                    
                    self.conv_2d3,
                    self.elu,
                    self.maxpool_2d3,
                    self.dropout_3)
        
        self.convnet4 = nn.Sequential (                    
                    self.conv_2d4,
                    self.elu,
                    self.maxpool_2d4,
                    self.dropout_4)

        self.fcn = nn.Sequential ( 
                    self.fc1,
                    self.elu,
                    self.dropout_5,
                    
                    self.fc2,
                    self.relu,
                    self.dropout_6,
                    
                    self.fc3)

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.convnet1 (x)
        # print (x.size())
        x = self.convnet2 (x)
        # print (x.size())
        x = self.convnet3 (x)
        # print (x.size())
        x = self.convnet4 (x)
        # print (x.size())

        x = x.view( (-1, x.shape[1] * x.shape[2]* x.shape[3] ))
        # print (x.size())

        x = self.fcn (x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
