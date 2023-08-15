import os 
from tqdm import tqdm

import torch 
import numpy as np

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from transformers import BertModel, BertConfig
from torch.nn.init import kaiming_normal_
from sklearn import metrics 
from models.utils import cal_acc_class


class SAKT(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            n: the length of the sequence of the question or responses
            d: the dimension of the hidden vectors in this model
            num_attn_heads: the number of the attention heads in the \
                multi-head attention module in this model
            dropout: the dropout rate of this model
    '''
    def __init__(self, num_q, n, d, num_attn_heads, dropout):
        super(SAKT, self).__init__()
        self.num_q = num_q
        self.n = n 
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2, self.d)
        self.E = Embedding(self.num_q, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        
        # BERT for feature extraction
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        self.at_emb_layer = Linear(768, self.d)
        self.at2_emb_layer = Linear(512, self.d)
        self.v_emb_layer = Linear(self.emb_size * 2, self.hidden_size)
        
        self.e_layer = Linear(self.hidden_size, self.emb_size)
        self.a_layer = Linear(self.hidden_size, self.emb_size)


        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout)
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, qry, at_s, at_t, at_m):
        '''
            Args: 
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                qry: the query sequence with the size of [batch_size, m], \
                    where the query is the question(KC) what the user wants \
                    to check the knowledge level of

            Returns:
            p: the knowledge level about the query
            attn_weights: the attention weights from the multi-head \
                attention module
        '''

        x =  q + self.num_q * r # 원문에서는 문제 + 응답 * 총 문제 수라고 함.

        M = self.M(x).permute(1, 0, 2)
        E = self.E(qry).permute(1, 0, 2) # E도 x로 이루어진 행렬이어야 함. 수정 필요
        P = self.P.unsqueeze(1)

        causal_mask = torch.triu(
            torch.ones([E.shape[0], M.shape[0]]),
            diagonal=1
        ).bool()
        
        # BERT, 양 차원 모양 바꾸기 
        A = self.at_emb_layer(self.bertmodel(input_ids=at_s,
                       attention_mask=at_t,
                       token_type_ids=at_m
                       ).last_hidden_state)
        A = self.at2_emb_layer(A.permute(0, 2, 1)) # 어텐션에 들어가는 형식으로 바까줌
        V = torch.relu(self.v_emb_layer(torch.concat([M, A], dim=-1)))
        print(V.shape, M.shape, P.shape)
        M = M + P + V
        
        # M = M + P

        S, attn_weights = self.attn(E, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        S = S.permute(1, 0, 2)
        M = M.permute(1, 0, 2)
        E = E.permute(1, 0, 2)

        S = self.attn_layer_norm(S + M + E) # Residual Connection and Layer normalization.
        


        # 주관식 문제도 포함 시키기, 포함시킨 뒤 Residual에도 추가
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S) # Residual Connection and Layer normalization.

        p = torch.sigmoid(self.pred(F)).squeeze(-1)

        return p, attn_weights
    
def sakt_train(model, train_loader, valid_loader, test_loader, num_q, num_epochs, batch_size, opt, ckpt_path):
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
    accs = []
    q_accs = {}
    
    max_auc = 0

    for i in tqdm(range(0, num_epochs)):
        loss_mean = []

        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()
            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y, _ = model(q.long(), r.long(), qshft_seqs.long(), bert_s, bert_t, bert_m)

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            y = torch.masked_select(y, m)
            t = torch.masked_select(rshft_seqs, m)

            opt.zero_grad()
            loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
            loss.backward()
            opt.step()
            
            loss_mean.append(loss.detach().cpu().numpy())
        auc = metrics.roc_auc_score(
            y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy()
        )
        bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
        acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)
        loss_mean = np.mean(loss_mean)

        print(f"[Train] number: {i}, AUC: {auc}, ACC: {acc} Loss Mean: {loss_mean}")
        # Validation
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()

                y, _ = model(q.long(), r.long(), qshft_seqs.long(), bert_s, bert_t, bert_m)

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )
                bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
                acc = metrics.accuracy_score(t.numpy(), bin_y)
                loss = binary_cross_entropy(y, t)
                
                print(f"[Valid] number: {i}, AUC: {auc}, ACC: {acc} Loss: {loss} ")

                if auc > max_auc : 
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    max_auc = auc
    # Test
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    with torch.no_grad():
        loss_mean = []
        for i, data in enumerate(test_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            model.eval()

            y, _ = model(q.long(), r.long(), qshft_seqs.long(), bert_s, bert_t, bert_m)

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            q = torch.masked_select(qshft_seqs, m).detach().cpu()
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(rshft_seqs, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
            acc = metrics.accuracy_score(t.numpy(), bin_y)
            loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

            
            print(f"[Test] number: {i}, AUC: {auc}, ACC: :{acc} Loss: {loss} ")

            # evaluation metrics
            aucs.append(auc)
            loss_mean.append(loss)     
            accs.append(acc)
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
        loss_means.append(np.mean(loss_mean))

    return aucs, loss_means, accs, q_accs, cnt