import os 
from tqdm import tqdm

import numpy as np 
import torch 

from torch.nn import Module, Parameter, Embedding, Linear, Dropout, TransformerEncoder, TransformerEncoderLayer, Sequential, LayerNorm, ReLU
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy, pad
from sklearn import metrics 
from transformers import BertModel, BertConfig
from models.utils import cal_acc_class


class BACK_DKVMN(Module):
    '''
        Args:
            num_q: the total number of the questions(KCs) in the given dataset
            dim_s: the dimension of the state vectors in this model
            size_m: the memory size of this model
    '''
    def __init__(self, num_q, dim_s, size_m) -> None:
        super(BACK_DKVMN, self).__init__()
        self.num_q = num_q 
        # 새로추가 됨
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
        self.d_emb_layer = Embedding(self.num_q, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # Right network
        # transformer network for feature extraction
        encoder_layers = TransformerEncoderLayer(d_model=self.dim_s, nhead=2)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=1)

        # BERT for feature extraction
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        self.at_emb_layer = Sequential(
            Linear(768, self.dim_s),
            ReLU(),
            LayerNorm(self.dim_s)
        )
        self.at_emb_layer = Linear(768, self.dim_s)
        self.at2_emb_layer = Linear(512, self.dim_s)

        self.qr_emb_layer = Embedding(2 * self.num_q, self.dim_s)

        self.v_emb_layer = Embedding(2 * self.num_q, self.dim_s)
        self.fusion_layer = Linear(2 * self.dim_s, self.dim_s)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

        # final network
        self.f_layer = Linear(2 * self.dim_s, self.dim_s)
        self.p_layer = Linear(2 * self.dim_s, 1)

        self.dropout_layer = Dropout(0.2)
        

    
    def forward(self, q, r, at_s, at_t, at_m, q2diff, qid):
        '''
            Args: 
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                at: the answer text sequence with the size of [batch_size, n]

            Returns:
                p: the knowledge level about q
                Mv: the value matrices from q, r, at
        '''
        x = q + self.num_q * r 
        batch_size = x.shape[0]

        # BERT를 사용하지 않는다면 주석처리
        em_at = self.at_emb_layer(self.bertmodel(input_ids=at_s,
                       attention_mask=at_t,
                       token_type_ids=at_m
                       ).last_hidden_state)
        em_at = self.at2_emb_layer(em_at.permute(0, 2, 1))

        # unsqueeze는 지정된 위치에 크기가 1인 텐서 생성 
        # repeat은 현재 갖고 있는 사이즈에 매개변수 만큼 곱해주는 것 (공간 생성, element가 있다면 해당 element 곱해줌.)
        # 그래서 Mv0 맨 앞에 차원 하나 추가하고 추가한 차원에 batch_size 곱해줌. 나머지는 메모리사이즈 * 1, 임베딩 공간 * 1 해줌.
        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1) # parameter로 메모리 사이즈만큼, 그리고 임베딩 공간만큼 구해줬음
        Mv = [Mvt] # 맨 첫번째 상태 만들어줌. shape: [1, batch size, memory_size, emb_size] 

        # 논문에서 봤던 대로 좌 우측 임베딩.
        k = self.k_emb_layer(q) # 보통의 키는 컨셉 수
        
        # BERT 사용 여부
        # v = self.v_emb_layer(q + r) 
        v = self.v_emb_layer(x) # 컨셉수, 응답 수.. 보통은 
        
        # Correlation Weight
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

        # Read Process 
        f = torch.tanh(
            self.f_layer(
            torch.cat(
                [
                    (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                    k
                ],
                dim=-1
            )
            )
        )
        
        f_l = torch.relu(self.fusion_layer(torch.concat([v, em_at], dim=-1)).permute(0, 2, 1))
        
        p = self.p_layer(torch.concat([self.dropout_layer(f), f_l], dim=-1))
        p = torch.sigmoid(p)
        p = p.squeeze(-1)

        return p, Mv, w
    
def train_model(model, train_loader, valid_loader, test_loader, num_q, num_epochs, batch_size, opt, ckpt_path):
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
    precisions = []
    recalls = []
    f1s = []
    
    max_auc = 0

    for i in tqdm(range(0, num_epochs)):
        loss_mean = []

        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()
            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y, Mv, w = model(q.long(), r.long(), bert_s, bert_t, bert_m, q2diff_seqs.long(), pid_seqs.long())

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            y = torch.masked_select(y, m)
            t = torch.masked_select(r, m)

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

        print(f"[Train] number: {i}, AUC: {auc}, ACC: {acc} ")
        # Validation
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()

                y, Mv, w = model(q.long(), r.long(), bert_s, bert_t, bert_m, q2diff_seqs.long(), pid_seqs.long())

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(r, m).detach().cpu()

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
        for i, data in enumerate(test_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            model.eval()

            y, Mv, w = model(q.long(), r.long(), bert_s, bert_t, bert_m, q2diff_seqs.long(), pid_seqs.long())

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            q = torch.masked_select(q, m).detach().cpu()
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(r, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
            acc = metrics.accuracy_score(t.numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

            print(f"[Test] number: {i}, AUC: {auc}, ACC: :{acc} Loss: {loss} ")

            # evaluation metrics
            aucs.append(auc)
            loss_mean.append(loss)     
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
        loss_means.append(np.mean(loss_mean))


    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s