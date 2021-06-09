import torch.nn as nn
import torch

class MyLSTM(nn.Module):

    def __init__(self,vocab,embed=100,hidden=128,layers=2,out=1,batch_first=True,dropout=0.6,device='cpu'):
        '''
        Initialize LSTM

        Args:
        vocab (int)       : vocabulary size
        embed (int)       : embedding layer dimension
        hidden (int)      : number of hidden units in LSTM layer
        layers (int)      : number of recurrent layers
        out (int)         : output dimension
        batch_first (bool): if batch is firstin dataset, 'True'
        dropout (float)   : dropout probability [0,1)
        '''
        super(MyLSTM,self).__init__()

        self.layers = layers
        self.batch_first = batch_first
        self.hidden = hidden
        self.device = device
        self.embed = nn.Embedding(vocab,embed)
        self.lstm = nn.LSTM(embed,hidden,num_layers=layers,dropout=dropout,batch_first=batch_first)
        self.fc = nn.Linear(hidden,out)
        self.drop = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()


    def __initiliaze(self,batch_size):
        '''
        Initializes hidden and cell state of first layer to zeroes

        Args:
        batch_size (int) : batch size 
        '''
        weights = next(self.parameters()).data
        if self.batch_first:
            hidden_states = (weights.new((self.layers,batch_size,self.hidden).zero.to(self.device)), weights.new((self.layers,batch_size,self.hidden).zero.to(self.device)))

        return hidden_states
    
    def forward(self,x,hidden):

        #Embed
        out = self.embed(x)

        #LSTM
        out, hidden = self.lstm(out,hidden)
        out = out.contigous().view(-1,self.hidden)

        #Dense
        out = self.fc(out)
        out = out.view(x.shape[0],-1)

        out = out[:,-1]

        return out, hidden

