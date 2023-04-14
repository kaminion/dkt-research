import os 
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics

class DKT(Module):
    '''
        Args: 
            num_q: total number of questions in given the dataset
    '''
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size) # log2M의 길이를 갖는 따르는 랜덤 가우시안 벡터에 할당하여 인코딩 (평균 0, 분산 I)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)
        self.dropout_layer = Dropout()

    def forward(self, q, r):
        '''
            q: the questions(KC) sequence with the size of [batch_size, n]
            r: response sequence with the size of [batch_size, n]
        '''

        # Compressive sensing에 의하면 d차원에서의 k-sparse 신호는 모두 원복될 수 있음. (klogd에 변형을 가한 모든 값)
        # 여기서 d차원은 unique exercise이고(M), K-sparse는 원핫인코딩을 거치므로 1-sparse라고 할 수 있음.
        x = q + self.num_q * r # r텐서를 num_q 만큼 곱해서 확장함
        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                test_loader: the PyTorch DataLoader instance for test
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        aucs = []
        loss_means = []  

        max_auc = 0

        for i in range(0, num_epochs):
            loss_mean = []

            for data in train_loader:
                q, r, qshift, rshift, m = data

                self.train()
                
                # 현재까지의 입력을 받은 뒤 다음 문제 예측
                y = self(q.long(), r.long())
                # one-hot은 sparse한 벡터를 아웃풋으로 가짐, 주어진 텐서의 값을 인덱스 번호로 보고 원 핫 임베딩을 함. 뒤에는 num_classes.
                # 예를 들어 tensor([0, 1, 2, 0, 1]) 인 경우, 인덱스 번호를 원핫으로 보고 
                # tensor([[1, 0, 0, 0, 0],
                # [0, 1, 0, 0, 0],
                # [0, 0, 1, 0, 0],
                # [1, 0, 0, 0, 0],
                # [0, 1, 0, 0, 0]]) 로 배출함
                # 여기서는 해당 문제임을 알기 위해 원 핫 사용
                y = (y * one_hot(qshift.long(), self.num_q)).sum(-1) # 이 과정이 원 핫, 곱해서 정답과 매핑시킴 
                
                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshift, m)

                opt.zero_grad()
                loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, qshift, rshift, m = data 

                    self.eval()

                    y = self(q.long(), r.long())
                    y = (y * one_hot(qshift.long(), self.num_q)).sum(-1)

                    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshift, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                    
                    print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean} ")

                    if auc > max_auc : 
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "model.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means