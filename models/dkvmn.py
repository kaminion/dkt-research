import os 

import numpy as np 
import torch 

from torch.nn import Module, Parameter, Embedding, Linear
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics 


class DKVMN(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
    def __init__(self, num_q, dim_s, size_m) -> None:
        super(DKVMN, self).__init__()
        self.num_q = num_q 
        self.dim_s = dim_s 
        self.size_m = size_m

        # M은 각 location의 벡터사이즈, N은 memory location의 수 
        # Mt: N*M matrix
        # 각 문제에는 N latent concepts가 존재한다고 가정
        # 각 가정된 컨셉은 key matrix Mk에 저장 (N * key의 dimension. N은 latent concepts)
        # 각 컨셉의 학생들의 마스터리 value matrix Mv에 저장 (N * value의 dimension)


        # Left network
        # q_t를 받는 부분, 논문상에서 A 과정, K_t라고 볼 수 있음
        self.k_emb_layer = Embedding(self.num_q, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # Right network
        self.v_emb_layer = Embedding(2 * self.num_q, self.dim_s)
        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

        # final network
        self.f_layer = Linear(2 * self.dim_s, self.dim_s)
        self.p_layer = Linear(self.dim_s, 1)

    
    def forward(self, q, r):
        '''
            Args: 
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]

            Returns:
                p: the knowledge level about q
                Mv: the value matrices from q, r
        '''

        x = q + self.num_q * r 

        batch_size = x.shape[0]
        # unsqueeze는 지정된 위치에 크기가 1인 텐서 생성 
        # repeat은 현재 갖고 있는 사이즈에 매개변수 만큼 곱해주는 것 (공간 생성, element가 있다면 해당 element 곱해줌.)
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1) # parameter로 메모리 사이즈만큼, 그리고 임베딩 공간만큼 구해줬음
        Mv = [Mvt]

        # 논문에서 봤던 대로 좌 우측 임베딩.
        k = self.k_emb_layer(q) # 보통의 키는 컨셉 수 
        v = self.v_emb_layer(x) # 컨셉수, 응답 수.. 보통은 
        
        # Correlation Weight
        w = torch.softmax(torch.matmul(k.T, self.Mk), dim=-1) # 차원이 세로로 감, 0, 1, 2 뎁스가 깊어질 수록 가로(행)에 가까워짐, 모든 row 데이터에 대해 softmax 

        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))
    