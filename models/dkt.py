import os 
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from models.utils import collate_fn, equalized_odd


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
        x = q + r # r텐서를 num_q 만큼 곱해서 확장함
        h, _ = self.lstm_layer(self.interaction_emb(x))
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y


def dkt_train(model, train_loader, test_loader, exp_loader, num_q, num_epochs, opt, ckpt_path):
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
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()

            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y = model(q.long(), r.long())
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

            opt.zero_grad()
            y = torch.masked_select(y, m)
            t = torch.masked_select(rshft_seqs, m)
            regularization, eq_odd = equalized_odd(y, t, q)

            loss = binary_cross_entropy(y, t) + regularization
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for data in test_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()
                
                y = model(q.long(), r.long())
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()
                _, eq_odd = equalized_odd(y, t, q)

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                
                if auc > max_auc : 
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    print(f"Epoch {i}, previous AUC: {max_auc}, max AUC: {auc}, eq_odd: {eq_odd}")
                    max_auc = auc

                loss_means.append(loss_mean)

    # 실제 성능측정
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    for i in range(1, num_epochs + 1):
        with torch.no_grad():
            for data in exp_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()
                y = model(q.long(), r.long())
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)
                _, eq_odd = equalized_odd(y, t, q)


                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                
                print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean}, eq_odd: {eq_odd}")

                aucs.append(auc)

    return aucs, loss_means