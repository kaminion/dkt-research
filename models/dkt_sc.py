import os
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MultiheadAttention, LayerNorm, ModuleList
from models.emb import STFTEmbedding
import torch.nn.functional as F
from torch.nn.functional import one_hot, binary_cross_entropy
from torch.autograd import Variable
from sklearn import metrics
from models.utils import calculate_dis_impact

from transformers import BertModel, BertConfig, DistilBertConfig, DistilBertModel
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
        self.x2h = Linear(self.input_size, 4 * self.hidden_size, bias=self.bias)
        self.h2h = Linear(self.hidden_size, 4 * self.hidden_size, bias=self.bias)
        self.reset_parameters()
        
    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
        
    def forward(self, x, hx):
        
        if hx is None:
            hx = Variable(x.new_zeros(x.size(0), self.hidden_size))
            hx = (hx, hx)
            
        hx, cx = hx
    
        gates = self.x2h(x) + self.h2h(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate) # 입력 게이트에 시그모이드 적용
        forgetgate = torch.sigmoid(forgetgate) # 망각 게이트에 시그모이드 적용
        cellgate = torch.tanh(cellgate) # 셀 게이트에 탄젠트 적용
        outgate = torch.sigmoid(outgate) # 출력 게이트에 시그모이드 적용
        
        cy = cx * forgetgate + ingate * cellgate
        hy = outgate * torch.tanh(cy)
                
        return (hy, cy)
    
    
class LSTMModel(Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, bias=True):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.bias = bias
        
        self.rnn_cell_list = ModuleList()
        self.rnn_cell_list.append(LSTMCell(self.input_dim, self.hidden_dim, self.bias))
        
        for l in range(1, layer_dim):
            self.rnn_cell_list.append(LSTMCell(self.input_dim, self.hidden_dim, self.bias))

        self.fc = Linear(self.hidden_dim, self.hidden_dim)
        
        # BERT를 위한 추가 레이어
        # bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        distilconfig = DistilBertConfig(output_hidden_states=True)
        self.bertmodel = DistilBertModel.from_pretrained('distilbert-base-uncased', config=distilconfig)
        
        self.at_emb_layer = Linear(768, self.hidden_dim)
        self.at2_emb_layer = Linear(512, self.hidden_dim)
        self.v_emb_layer = Linear(self.hidden_dim * 2, self.hidden_dim)
        
    def forward(self, x, at_s, at_t, at_m):
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        outs = []

        # 여기 BERT 추가해서 돌림
        # BERT, 양 차원 모양 바꾸기 
        # at = self.at_emb_layer(self.bertmodel(input_ids=at_s,
        #                attention_mask=at_t,
        #                token_type_ids=at_m
        #                ).last_hidden_state)
        bt = self.bertmodel(input_ids=at_s, attention_mask=at_t)
        at = self.at_emb_layer(bt.last_hidden_state)
        at = self.at2_emb_layer(at.permute(0, 2, 1)) # 6, 100, 100 형태로 바꿔줌.
        v = torch.relu(self.v_emb_layer(torch.concat([x, at], dim=-1)))

        hidden = list()
        for layer in range(self.layer_dim):
            hidden.append((h0[layer, :, :], h0[layer, :, :]))
            
        for t in range(x.size(1)):
            for layer in range(self.layer_dim):
                if layer == 0:
                    hidden_l = self.rnn_cell_list[layer](v[:, t, :], (hidden[layer][0], hidden[layer][1]))
                else:
                    hidden_l = self.rnn_cell_list[layer](hidden[layer - 1][0], (hidden[layer][0], hidden[layer][1]))
                hidden[layer] = hidden_l
            outs.append(hidden_l[0])
        
        out = outs[-1].squeeze() #.squeeze() 제외
        out = self.fc(out)
        return out


class DKT_FUSION(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT_FUSION, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size) # log2M의 길이를 갖는 따르는 랜덤 가우시안 벡터에 할당하여 인코딩 (평균 0, 분산 I)
        self.lstm_layer = LSTMModel(
            self.emb_size, self.hidden_size, 1, bias=True # concat 시 emb_size * 2
        )
        self.out_layer = Linear(self.hidden_size, self.num_q) # 원래 * 2이었으나 축소
        self.dropout_layer = Dropout()
        
        
    def forward(self, q, r, at_s, at_t, at_m):
        x = self.interaction_emb(q + self.num_q * r) # r텐서를 num_q 만큼 곱해서 확장함
        
        h = self.lstm_layer(x, at_s, at_t, at_m)
        
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y
    

def dkt_train(model, train_loader, valid_loader, test_loader, num_q, num_epochs, batch_size, opt, ckpt_path):
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
    disparate_impacts = []
    accs = []
    q_accs = {}
    precisions = []
    recalls = []
    f1s = []
    
    max_auc = 0

    for i in range(0, num_epochs):
        loss_mean = []

        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()

            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y = model(q.long(), r.long(), bert_s, bert_t, bert_m) # r 대신 pid_seq
            print(qshft_seqs.shape, y.shape, num_q, one_hot(qshft_seqs.long(), num_q).shape)
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

            opt.zero_grad()
            y = torch.masked_select(y, m)
            t = torch.masked_select(pidshift, m) # rshft 대신 pidshift
            h = torch.masked_select(hint_seqs, m)

            regularization, dis_impact = calculate_dis_impact(y, t, h)

            loss = binary_cross_entropy(y, t) 
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())
        auc = metrics.roc_auc_score(
            y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy()
        )
        bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
        acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)
        loss_mean = np.mean(loss_mean)

        print(f"[Train] epoch: {i}, AUC: {auc}, acc: {acc}, Loss Mean: {np.mean(loss_mean)}")

        with torch.no_grad():
            loss_mean = []
            for i, data in enumerate(valid_loader):
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()
                
                y = model(q.long(), pid_seqs.long(), bert_s, bert_t, bert_m)
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(pidshift, m).detach().cpu()
                h = torch.masked_select(hint_seqs, m).detach().cpu()

                non_roc, sen_roc = calculate_dis_impact(y, t, h)

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )
                bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
                acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)

                loss = binary_cross_entropy(y, t) 
                print(f"[Valid] number: {i}, AUC: {auc}, ACC: {acc}, loss: {loss}")

                if auc > max_auc : 
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    max_auc = auc


    # 실제 성능측정
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    loss_mean = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            model.eval()
            y = model(q.long(), pid_seqs.long(), bert_s, bert_t, bert_m)
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)


            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            q = torch.masked_select(q, m).detach().cpu()
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(pidshift, m).detach().cpu()
            h = torch.masked_select(hint_seqs, m).detach().cpu()

            _, dis_impact = calculate_dis_impact(y, t, h)

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
            acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            loss = binary_cross_entropy(y, t) 

            
            print(f"[Test] number: {i}, AUC: {auc}, ACC: {acc}, loss: {loss}")

            aucs.append(auc)
            disparate_impacts.append(dis_impact.detach().cpu())
            loss_mean.append(loss)
            accs.append(acc)
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
        loss_means = np.mean(loss_mean) # 실제 로스 평균값을 구함

    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s