import os 
import numpy as np 
import torch
from torch import nn 
from torch.nn.functional import binary_cross_entropy, softmax, binary_cross_entropy_with_logits
from sklearn import metrics

class DeepIRT(nn.Module):
    def __init__(self, num_q, num_u, emb_size, hidden_size):
        super(DeepIRT, self).__init__()
        self.num_q = num_q
        self.num_u = num_u
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.user_emb = nn.Embedding(self.num_q * self.num_u, self.emb_size)
        self.item_emb = nn.Embedding(2 * self.num_q, self.emb_size)

        self.examinee_layer = nn.Sequential()
        self.examinee_layer.add_module("theta1", nn.Linear(self.emb_size, self.hidden_size))
        self.examinee_layer.add_module("theta1_act", nn.Tanh())
        self.examinee_layer.add_module("theta2", nn.Linear(self.hidden_size, 1))
        self.examinee_layer.add_module("theta2_act", nn.Tanh())
        self.examinee_layer.add_module("theta3", nn.Linear(1, 1))

        self.item_layer = nn.Sequential()
        self.item_layer.add_module("beta1", nn.Linear(self.emb_size, self.hidden_size))
        self.item_layer.add_module("beta1_act", nn.Tanh())
        self.item_layer.add_module("beta2", nn.Linear(self.hidden_size, 1))
        self.item_layer.add_module("beta2_act", nn.Tanh())
        self.item_layer.add_module("beta3", nn.Linear(1, 1))

        self.hidden_layer = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()
        self.softmax = softmax


    def forward(self, q, r):
        emb_user = self.user_emb(q)
        emb_item = self.item_emb(q + r)

        theta = self.examinee_layer(emb_user)
        beta = self.item_layer(emb_item)
        h = self.hidden_layer(theta - beta)
        return self.softmax(h.squeeze(), dim=1)
    
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
                q, r, _, _, m = data

                self.train()
                
                # 현재까지의 입력을 받은 뒤 다음 문제 예측
                y = self(q.long(), r.long())
                y = torch.masked_select(y, m)
                t = torch.masked_select(r, m)

                opt.zero_grad()
                loss = binary_cross_entropy_with_logits(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, _, _, m = data 

                    self.eval()

                    y = self(q.long(), r.long())
                    # y = (y * one_hot(qshift.long(), self.num_q)).sum(-1)

                    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(r, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                    
                    print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean} ")

                    if auc > max_auc : 
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "dirt.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means