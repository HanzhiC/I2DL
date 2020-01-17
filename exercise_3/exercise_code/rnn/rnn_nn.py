import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, activation="tanh"):
        super().__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        #######################################################################
        # TODO: Build a simple one layer RNN with an activation with the      #
        # attributes defined above and a forward function below. Use the      #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h as 0 if these values are not given.                     #
        #######################################################################
        self.hidden_size = hidden_size
        self.input_size = input_size
        
        self.w = nn.Linear (hidden_size, hidden_size)
        self.v = nn.Linear (input_size, hidden_size)
        self.b = torch.zeros (hidden_size)
        # self.bh = torch.zeros ()
        self.activation = nn.Tanh()
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        h_seq = torch.zeros (x.shape[0]+1, x.shape[1], self.hidden_size)
        seq_len = x.shape[0]
        # print ("# of sequence", seq_len)
        h_init = torch.zeros (1, x.shape[1], self.hidden_size)
        for i in range(seq_len):
            if i == 0:
                h_seq[0,:,:] = self.activation (self.w(h_init) +  self.v(x[0,:,:]) + self.b )
            else:
                h_seq[i,:,:] = self.activation (self.w(h_seq[i-1,:,:]) +  self.v(x[i,:,:]) + self.b )
            # print (h_seq[i,:,:])
        h_seq = h_seq [0:x.shape[0],:,:]
        h = h_seq[-1]
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, h


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=20):
        super().__init__()
        #######################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes #
        # defined above and a forward function below. Use the                 #
        # nn.Linear() function as your linear layers.                         #
        # Initialse h and c as 0 if these values are not given.               #
        #######################################################################
        self.catsize = input_size + hidden_size 
        self.hidden_size = hidden_size
        
        self.gate = nn.Sigmoid()
        self.act = nn.Tanh()
        
        self.uf = nn.Linear (hidden_size, hidden_size)
        self.wf = nn.Linear (input_size, hidden_size) 
        self.bf = torch.zeros (hidden_size) 

        self.ui = nn.Linear (hidden_size, hidden_size)
        self.wi = nn.Linear (input_size, hidden_size) 
        self.bi = torch.zeros (hidden_size) 

        self.uo = nn.Linear (hidden_size, hidden_size)
        self.wo = nn.Linear (input_size, hidden_size) 
        self.bo = torch.zeros (hidden_size) 

        self.uc = nn.Linear (hidden_size, hidden_size)
        self.wc = nn.Linear (input_size, hidden_size) 
        self.bc = torch.zeros (hidden_size) 

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x, h=None, c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence
                 (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = None
        #######################################################################
        #                                YOUR CODE                            #
        #######################################################################
        h_seq = torch.zeros (x.shape[0]+1, x.shape[1], self.hidden_size)
        c_seq = torch.zeros (x.shape[0]+1, x.shape[1], self.hidden_size)
        h_init = torch.zeros (1, x.shape[1], self.hidden_size)

        seq_len = x.shape[0]

        for i in range(seq_len):
            h = h_seq[i-1] if i > 0 else h_init
            ft = self.gate (self.wf(x[i,:,:]) + self.uf (h) + self.bf)
            it = self.gate (self.wi(x[i,:,:]) + self.ui (h) + self.bi)
            ot = self.gate (self.wo(x[i,:,:]) + self.uo (h) + self.bo)
            c_pt = self.act (self.wc(x[i,:,:]) + self.uc (h) + self.bc)
            c_seq[i,:,:] = ft * c_seq[i-1,:,:] + it * c_pt
            h_seq[i,:,:] = ot * self.act (c_seq[i,:,:])
    

        h_seq = h_seq [0:x.shape[0],:,:]
        c_seq = c_seq [0:x.shape[0],:,:]

        h = h_seq[-1]
        c = c_seq[-1]
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return h_seq, (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128,
                 activation="relu"):
        super(RNN_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a RNN classifier                                       #
        #######################################################################
        self.hidden_size = hidden_size
        self.RNN = nn.RNN(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, classes)
        self.bn = nn.BatchNorm1d (64)
    def forward(self, x):
        batch_size = x.size()[1]
        _, x = self.RNN(x)
        x = F.relu(self.fc1(x.reshape(batch_size, self.hidden_size)))
        x = self.bn(x)
        x = self.fc2(x)
        return x

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


class LSTM_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28, hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        #######################################################################
        #  TODO: Build a LSTM classifier                                      #
        #######################################################################
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTM(input_size, hidden_size)

        self.fc1 = nn.Linear(hidden_size, 64)
        self.bn1 = nn.BatchNorm1d (64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d (32)
        self.fc3 = nn.Linear (32, classes)    

    def forward(self, x):
        batch_size = x.size()[1]
        _, (x, _) = self.LSTM(x)
        x = F.relu(self.fc1(x.reshape(batch_size, self.hidden_size)))
        x = self.bn1(x)
        x = F.relu(self.fc2(x.reshape(batch_size, 64)))
        x = self.bn2(x)      
        x = self.fc3(x)
        return x

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
