import os 
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MultiheadAttention, LayerNorm
from models.emb import STFTEmbedding
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from models.utils import calculate_dis_impact

from transformers import BertModel, BertConfig, DistilBertConfig, DistilBertModel
from models.utils import cal_acc_class


class DKT(Module):
    '''
        Args: 
            num_q: total number of questions in given the dataset
    '''
    def __init__(self, num_q, emb_size, hidden_size, dropout=0.2):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # BERT for feature extraction
        # bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        distilconfig = DistilBertConfig(output_hidden_states=True)
        self.bertmodel = DistilBertModel.from_pretrained('distilbert-base-uncased', config=distilconfig)
        self.at_emb_layer = Linear(768, self.emb_size)
        self.at2_emb_layer = Linear(512, self.emb_size)
        
        self.v_emb_layer = Linear(self.emb_size * 2, self.hidden_size)
        
        self.e_layer = Linear(self.hidden_size, self.emb_size)
        self.a_layer = Linear(self.hidden_size, self.emb_size)

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size) # log2M의 길이를 갖는 따르는 랜덤 가우시안 벡터에 할당하여 인코딩 (평균 0, 분산 I)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True # concat 시 emb_size * 2
        )
        self.out_layer = Linear(self.hidden_size, self.num_q) # 원래 * 2이었으나 축소
        self.dropout_layer = Dropout(self.dropout)


    def forward(self, q, r, at_s, at_t, at_m):
        '''
            q: the questions(KC) sequence with the size of [batch_size, n]
            r: response sequence with the size of [batch_size, n]
        '''

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
        # e = torch.sigmoid(self.e_layer(v))
        # a = torch.tanh(self.a_layer(v))
        
        h, _ = self.lstm_layer(x)
        
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y


def dkt_train(model, train_loader, valid_loader, num_q, num_epochs, fold_num, opt, ckpt_path):
    '''
        Args:
            train_loader: the PyTorch DataLoader instance for training
            test_loader: the PyTorch DataLoader instance for test
            num_epochs: the number of epochs
            opt: the optimization to train this model
            ckpt_path: the path to save this model's parameters
    '''
    # aucs = []
    # loss_means = []  
    # disparate_impacts = []
    # accs = []
    # q_accs = {}
    # precisions = []
    # recalls = []
    # f1s = []
    
    max_auc = 0

    for epoch in range(0, num_epochs):
        loss_mean = []

        for i, data in enumerate(train_loader, 0):
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()

            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y = model(q.long(), r.long(), bert_s, bert_t, bert_m) # r 대신 pid_seq
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

            opt.zero_grad()
            y = torch.masked_select(y, m)
            t = torch.masked_select(rshft_seqs, m) # rshft 대신 pidshift
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

        print(f"[Train] Epoch: {epoch}, AUC: {auc}, acc: {acc}, Loss Mean: {np.mean(loss_mean)}")

        with torch.no_grad():
            loss_mean = []
            for i, data in enumerate(valid_loader):
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()
                
                y = model(q.long(), r.long(), bert_s, bert_t, bert_m)
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()
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
        print(f"========== Finished Epoch: {epoch} ============")

    # # 실제 성능측정
    # model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    # loss_mean = []
    # with torch.no_grad():
    #     for i, data in enumerate(test_loader, 0):
    #         q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

    #         model.eval()
    #         y = model(q.long(), r.long(), bert_s, bert_t, bert_m)
    #         y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

    #         # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    #         q = torch.masked_select(q, m).detach().cpu()
    #         y = torch.masked_select(y, m).detach().cpu()
    #         t = torch.masked_select(rshft_seqs, m).detach().cpu()
    #         h = torch.masked_select(hint_seqs, m).detach().cpu()

    #         _, dis_impact = calculate_dis_impact(y, t, h)

    #         auc = metrics.roc_auc_score(
    #             y_true=t.numpy(), y_score=y.numpy()
    #         )
    #         bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
    #         acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)
    #         precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
    #         recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
    #         f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
    #         loss = binary_cross_entropy(y, t) 

            
    #         print(f"[Test] number: {i}, AUC: {auc}, ACC: {acc}, loss: {loss}")

    #         aucs.append(auc)
    #         disparate_impacts.append(dis_impact.detach().cpu())
    #         loss_mean.append(loss)
    #         accs.append(acc)
    #         q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
    #         precisions.append(precision)
    #         recalls.append(recall)
    #         f1s.append(f1)
            
    #     loss_means = np.mean(loss_mean) # 실제 로스 평균값을 구함

    # return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s