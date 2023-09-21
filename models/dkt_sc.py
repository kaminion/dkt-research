import os
import math
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MultiheadAttention, LayerNorm
from models.emb import STFTEmbedding
import torch.nn.functional as F
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.autograd import Variable
from sklearn import metrics
from models.utils import calculate_dis_impact

from transformers import BertModel, BertConfig
from models.utils import cal_acc_class



"""
각 단계에 대한 은닉 노드를 의미함. (hidden node), 총합을 사용하여 셀 값 반영, 기울기 사라짐 방지
"""
class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = Linear(input_size, 4 * hidden_size, bias=bias)
        self.h2h = Linear(hidden_size, 4 * hidden_size, bias=bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
        
    def forward(self, x, hidden):
        hx, cx = hidden
        x = x.view(-1, x.size(1))
        
        gates = self.x2h(x) + self.h2h(hx)
        gates = gates.squeeze()
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = F.sigmoid(ingate) # 입력 게이트에 시그모이드 적용
        forgetgate = F.sigmoid(forgetgate) # 망각 게이트에 시그모이드 적용
        cellgate = F.tanh(cellgate) # 셀 게이트에 탄젠트 적용
        outgate = F.sigmoid(outgate) # 출력 게이트에 시그모이드 적용
        
        cy = torch.mul(cx, forgetgate) + torch.mul(ingate, cellgate)
        hy = torch.mul(outgate, F.tanh(cy))
        
        return (hy, cy)
    
    
class LSTMModel(Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.bias = bias
        
        self.lstm = LSTMCell(self.input_dim, self.hidden_dim, self.bias)
        self.fc = Linear(self.hidden_dim, self.output_dim)
        
    def forward(self, x):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        outs = []
        cn = c0[0, :, :]
        hn = h0[0, :, :]
        
        for seq in range(x.size(1)):
            hn, cn = self.lstm(x[:, seq, :], (hn, cn))
            outs.append(hn)
        
        out = outs[-1].squeeze()
        out = self.fc(out)
        return out
    
class DKT_FUSION(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        
        # BERT for feature extraction
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        self.at_emb_layer = Linear(768, self.emb_size)
        self.at2_emb_layer = Linear(512, self.emb_size)
        
        self.v_emb_layer = Linear(self.emb_size * 2, self.hidden_size)

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size) # log2M의 길이를 갖는 따르는 랜덤 가우시안 벡터에 할당하여 인코딩 (평균 0, 분산 I)
        self.lstm_layer = LSTMModel(
            self.emb_size, self.hidden_size, bias=True # concat 시 emb_size * 2
        )
        self.out_layer = Linear(self.hidden_size, self.num_q) # 원래 * 2이었으나 축소
        self.dropout_layer = Dropout()
        
        
    def forward(self, q, r, at_s, at_t, at_m):
        # Compressive sensing에 의하면 d차원에서의 k-sparse 신호는 모두 원복될 수 있음. (klogd에 변형을 가한 모든 값)
        # 여기서 d차원은 unique exercise이고(M), K-sparse는 원핫인코딩을 거치므로 1-sparse라고 할 수 있음.
        x = self.interaction_emb(q + self.num_q * r) # r텐서를 num_q 만큼 곱해서 확장함
        
        # 여기 BERT 추가해서 돌림
        # BERT, 양 차원 모양 바꾸기 
        # at = self.at_emb_layer(self.bertmodel(input_ids=at_s,
        #                attention_mask=at_t,
        #                token_type_ids=at_m
        #                ).last_hidden_state)
        # at = self.at2_emb_layer(at.permute(0, 2, 1)) # 6, 100, 100 형태로 바꿔줌.

        # v = torch.relu(self.v_emb_layer(torch.concat([x, at], dim=-1)))
        
        h, _ = self.lstm_layer(x)
        
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y