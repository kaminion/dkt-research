import os 
import time
from tqdm import tqdm

import numpy as np 
import torch 

from torch.nn import Module, Parameter, Embedding, Linear, Dropout, GRU, MultiheadAttention, Sequential, LayerNorm, ReLU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy, pad
from sklearn import metrics 
from transformers import BertModel, BertConfig, DistilBertConfig, DistilBertModel
from models.utils import bert_tokenizer, cal_acc_class, mean_eval, mean_eval_ext, log_auc, save_auc_bert, early_stopping, common_append, val_append, dkvmn_bert_train, dkvmn_bert_train_csedm, dkvmn_bert_test, dkvmn_bert_test_csedm

import wandb

class SUBJ_DKVMN(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
    def __init__(self, num_q, dim_s, size_m, dropout) -> None:
        super(SUBJ_DKVMN, self).__init__()
        self.num_q = num_q 
        # 새로추가 됨
        self.dim_s = dim_s 
        self.size_m = size_m
        self.dropout = dropout

        # M은 각 location의 벡터사이즈, N은 memory location의 수 
        # Mt: N*M matrix
        # 각 문제에는 N latent concepts가 존재한다고 가정
        # 각 가정된 컨셉은 key matrix Mk에 저장 (N * key의 dimension. N은 latent concepts)
        # 각 컨셉의 학생들의 마스터리 value matrix Mv에 저장 (N * value의 dimension)


        # Left network
        # q_t를 받는 부분, 논문상에서 A 과정, K_t라고 볼 수 있음
        self.k_emb_layer = Embedding(self.num_q, self.dim_s)
        self.d_emb_layer = Embedding(self.num_q, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # Right network
        # transformer network for feature extraction
        # encoder_layers = TransformerEncoderLayer(d_model=self.dim_s, nhead=2)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # BERT for feature extraction
        # bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        distilconfig = DistilBertConfig(output_hidden_states=True)
        self.bertmodel = DistilBertModel.from_pretrained('bert-base-uncased', config=distilconfig)
        # self.bertmodel = DistilBertModel(config=distilconfig)
        self.bertmodel.resize_token_embeddings(len(bert_tokenizer))
        
        # BERT output dimension: 768
        self.wb = Linear(768, self.dim_s)
        self.wq = Linear(self.dim_s, self.dim_s)
        self.wk = Linear(self.dim_s, self.dim_s)
        self.wv = Linear(self.dim_s, self.dim_s)
        
        self.P = Parameter(torch.Tensor(200, self.dim_s))
        kaiming_normal_(self.P)
        
        self.attn = MultiheadAttention(self.dim_s, 5, self.dropout)
        self.attn_norm = LayerNorm(self.dim_s)
        self.attn_dropout = Dropout(self.dropout)
        
        # self.at_emb_layer = Sequential(
        #     Linear(768, self.dim_s),
        #     ReLU(),
        #     LayerNorm(self.dim_s)
        # )
        # self.at_emb_layer = Linear(768, self.dim_s)
        # self.at_emb_layer = Linear(512, self.dim_s)

        # 버트 허용여부
        self.v_emb_layer = Embedding(2 * self.num_q, self.dim_s)
        self.v_fusion_layer = Linear(2 * self.dim_s, self.dim_s)
        # self.v_emb_layer = Linear(2 * self.dim_s, self.dim_s)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)
        
        # 마지막에 버트 인코더로 한번 더 평가
        # distilconfig_2 = DistilBertConfig(output_hidden_states=True)
        # self.bertmodel_2 = DistilBertModel.from_pretrained('bert-base-uncased', config=distilconfig_2)
        # # self.bertmodel = DistilBertModel(config=distilconfig)
        # self.bertmodel_2.resize_token_embeddings(len(bert_tokenizer))
        # self.attn_2 = MultiheadAttention(self.dim_s, 5, self.dropout)
        # self.attn_norm_2 = LayerNorm(self.dim_s)

        # final network
        self.f_layer = Linear(2 * self.dim_s, self.dim_s)
        # self.bert_linear = GRU(768, self.dim_s, batch_first=True)
        
        self.FFN = Sequential(
            Linear(self.dim_s, self.dim_s),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.dim_s, self.dim_s),
            Dropout(self.dropout)
        )
        self.FFN_layer_norm = LayerNorm(self.dim_s)
        
        # self.fusion_layer2 = Linear(2 * self.dim_s, self.dim_s)
        # self.fusion_norm2 = LayerNorm(self.dim_s, self.dim_s)
        self.p_layer = Linear(self.dim_s, 1)

        self.dropout_layer = Dropout(self.dropout)
        

    
    def forward(self, q, r, at_s, at_t, at_m): # q2diff, qid 삭제
        '''
            Args: 
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                at: the answer text sequence with the size of [batch_size, n]

            Returns:
                p: the knowledge level about q
                Mv: the value matrices from q, r, at
        '''
        # x = q + r * self.num_q
        x = q + r * self.num_q
        
        batch_size = x.shape[0]
        
        # print(f"BERT_ids shape: {at_s.shape}")


        # unsqueeze는 지정된 위치에 크기가 1인 텐서 생성 
        # repeat은 현재 갖고 있는 사이즈에 매개변수 만큼 곱해주는 것 (공간 생성, element가 있다면 해당 element 곱해줌.)
        # 그래서 Mv0 맨 앞에 차원 하나 추가하고 추가한 차원에 batch_size 곱해줌. 나머지는 메모리사이즈 * 1, 임베딩 공간 * 1 해줌.
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1) # parameter로 메모리 사이즈만큼, 그리고 임베딩 공간만큼 구해줬음
        Mv = [Mvt] # 맨 첫번째 상태 만들어줌. shape: [1, batch size, memory_size, emb_size] 

        # 논문에서 봤던 대로 좌 우측 임베딩.
        k = self.k_emb_layer(q) # 보통의 키는 컨셉 수
        v = self.v_emb_layer(x)
        
        em_at = self.bertmodel(input_ids=at_s,
                attention_mask=at_m,
            #    token_type_ids=at_t
                ).last_hidden_state
        em_at = self.wb(em_at)
        
        v = self.v_fusion_layer(torch.concat([em_at, v], dim=-1))
        
        # Correlation Weight - 여기 S 원래는 k였음
        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1) # 차원이 세로로 감, 0, 1, 2 뎁스가 깊어질 수록 가로(행)에 가까워짐, 모든 row 데이터에 대해 softmax 
        # Write Process
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        # [100, 100, 20] [100, 100, 50] [100, 100, 50]
        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)
        
        Mv = torch.stack(Mv, dim=1)

        # diff = self.d_emb_layer(q2diff)
        # Read Process 
        f = torch.tanh(
            self.f_layer(
            torch.cat(
                [
                    (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                    k # + diff 삭제
                ],
                dim=-1
            )
            )
        )
        f = self.dropout_layer(f)
        
        P = self.P.unsqueeze(1)
        # SAKT처럼 활용하기 // 
        
        # key + answer text
        vq = self.wq(f).permute(1, 0, 2) + P
        vk = self.wk(v).permute(1, 0, 2)
        vv = self.wv(v).permute(1, 0, 2) 
        
        causal_mask = torch.triu(
            torch.ones([vq.shape[0], vk.shape[0]]),
            diagonal=1
        ).bool()
                
        S, attn_weight = self.attn(vq, vk, vv, attn_mask=causal_mask)
        S = self.attn_dropout(S)
        
        # 
        S = S.permute(1, 0, 2)
        vq = vq.permute(1, 0, 2)
        vk = vk.permute(1, 0, 2)
        vv = vv.permute(1, 0, 2)
        
        S = self.attn_norm(S + vq + vk + vv)
        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)
        
        # BERT 2번째, input이랑 
        # em_at2, _ = self.bert_linear(self.bertmodel_2(input_ids=at_s,
        #                attention_mask=at_m,
        #             #    token_type_ids=at_t
        #                ).last_hidden_state)

        
        # fula2 = self.fusion_layer2(torch.concat([f, em_at2], dim=-1))
        # f = self.fusion_norm2(f + em_at2 + fula2)
              
        # abil = torch.tanh(self.fusion_layer(f + em_at))
        
        # em_at = self.fusion_norm(f + em_at + abil)
        
        p = self.p_layer(F) # original = f

        p = torch.sigmoid(p)
        p = p.squeeze(-1)

        return p, Mv
    
    
def train_model(model, train_loader, valid_loader, num_q, num_epochs, opt, ckpt_path, mode=0, use_wandb=False):
    '''
        Args:
            train_loader: the PyTorch DataLoader instance for training
            test_loader: the PyTorch DataLoader instance for test
            num_epochs: the number of epochs
            opt: the optimization to train this model
            ckpt_path: the path to save this model's parameters
    '''
    max_auc = 0
    loss_means = []  
    aucs = []
    accs = []
    precisions = []
    q_accs = {}
    recalls = []
    f1s = []
    
    wandb_dict = {}
    if use_wandb == True: 
        wandb_dict = {
                    "dropout": wandb.config.dropout, 
                    "lr": wandb.config.learning_rate,
                    "size_m": wandb.config.size_m,
                    "dim_s": wandb.config.dim_s
                }
    
        # "n": 50,
        # "d": 512,
        # "num_attn_heads": 4,
        # "num_tr_layers": 4,
        # "dropout": 0.1
        
    for epoch in range(0, num_epochs):
        auc_mean = []
        loss_mean = []
        acc_mean = []
        
        model.train()
        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            
            # CSEDM에선 PID_SEQS 대신 LABEL_SEQ로 취급함. 
            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y, t, loss = None, None, None
            
            if mode == 1: # CSEDM
                y, t, loss = dkvmn_bert_train_csedm(model, opt, q, r, bert_s, bert_t, bert_m, q2diff_seqs, m)
            else:
                y, t, loss = dkvmn_bert_train(model, opt, q, r, bert_s, bert_t, bert_m, m)

            common_append(y, t, loss, loss_mean, auc_mean, acc_mean)
            
        loss_mean, auc_mean, acc_mean =  mean_eval(loss_mean, auc_mean, acc_mean)

        log_auc(use_wandb, {
            "epoch": epoch,
            "train_auc": auc_mean,
            "train_acc": acc_mean,
            "train_loss": loss_mean
        })

        if(epoch % 10 == 0):
            print(f"[Train] Epoch: {epoch}, AUC: {auc_mean}, acc: {acc_mean}, Loss Mean: {loss_mean}")

        # Validation
        model.eval()
        with torch.no_grad():
            auc_mean = []
            loss_mean = []
            acc_mean = []
            precision_mean = []
            recall_mean = []
            f1_mean = []
            
            # best_loss = 10 ** 9
            # patience_limit = 3
            # patience_check = 0

            for data in valid_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
                
                y, t, loss = None, None, None
                
                if mode == 1: # CSEDM
                    q, y, t, loss, Mv = dkvmn_bert_test_csedm(model, q, r, bert_s, bert_t, bert_m, q2diff_seqs, m)
                else:
                    q, y, t, loss, Mv = dkvmn_bert_test(model, q, r, bert_s, bert_t, bert_m, m)
                                
                # patience_check = early_stopping(best_loss, loss, patience_check)
                # if(patience_check >= patience_limit):
                #     break
                
                try:
                    auc = metrics.roc_auc_score(y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy())
                except:
                    continue
                
                max_auc = save_auc_bert(model, max_auc, auc, \
                            wandb_dict, # 만약 wandb 체크 안했다면 빈 dict 들어감
                            ckpt_path, use_wandb)
                bin_y = common_append(y, t, loss, loss_mean, auc_mean, acc_mean)
                val_append(t, bin_y, precision_mean, recall_mean, f1_mean)
                q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
                
            loss_mean, auc_mean, acc_mean =  mean_eval(loss_mean, auc_mean, acc_mean)
            precision_mean, recall_mean, f1_mean = mean_eval_ext(precision_mean, recall_mean, f1_mean)

            log_auc(use_wandb, {
                "epoch": epoch,
                "val_auc": auc_mean,
                "val_acc": acc_mean,
                "val_loss": loss_mean
            })
            
            aucs.append(auc_mean)
            loss_means.append(loss_mean)
            accs.append(acc_mean)
            precisions.append(precision_mean)
            recalls.append(recall_mean)
            f1s.append(f1_mean)
            if epoch % 10 == 0:
                print(f"[Valid] Epoch: {epoch} Result: AUC: {auc_mean}, ACC: {acc_mean}, loss: {loss_mean}")
            
        print(f"========== Finished Epoch: {epoch} ============")
    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s


def test_model(model, test_loader, num_q, ckpt_path, mode, use_wandb):
    loss_means = []
    aucs = []
    accs = []
    precisions = []
    q_accs = {}
    recalls = []
    f1s = []
    # Test
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            y, t, loss = None, None, None
                
            if mode == 1: # CSEDM
                q, y, t, loss, Mv = dkvmn_bert_test_csedm(model, q, r, bert_s, bert_t, bert_m, q2diff_seqs, m)
            else:
                q, y, t, loss, Mv = dkvmn_bert_test(model, q, r, bert_s, bert_t, bert_m, m)
            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
            acc = metrics.accuracy_score(t.numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            print(f"[Test] number: {i}, AUC: {auc}, ACC: :{acc} Loss: {loss} ")

            # evaluation metrics
            aucs.append(auc)
            loss_means.append(loss)     
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
        loss_means.append(np.mean(loss_means))
    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s
