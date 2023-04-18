import os 
import numpy as np
import torch
from torch import nn
from torch.nn import Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics


class QAKT(nn.Module):
    def __init__(self, num_q, emb_size, hidden_size) :
        super(QAKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = nn.Embedding(2 * self.num_q, self.emb_size)



    def forward(self, q, r):
        x = q + self.num_q * r 
        x = self.interaction_emb(x)