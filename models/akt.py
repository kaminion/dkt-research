import os
import torch 
from torch import nn
from torch.nn import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math 
import torch.nn.functional as F 
from enum import IntEnum
import numpy as np

from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics 
from models.utils import cal_acc_class


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class AKT(nn.Module):
    '''
        Args:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff: dimension for fully connected net inside the basic block
    '''
    def __init__(self, n_question, n_pid, d_model=256, n_blocks=1,
                 kq_same=True, dropout=0.05, model_type='akt', final_fc_dim=512, n_heads=8, d_ff=256, l2=1e-5,
                 separate_qa=False
                ):
        super(AKT, self).__init__()

        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = model_type
        self.separate_qa = separate_qa
        embed_l = d_model

        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)
        # n_question + 1, d_model
        self.q_embed = nn.Embedding(self.n_question + 1, embed_l)
        if self.separate_qa:
            self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
        else:
            self.qa_embed = nn.Embedding(2, embed_l)
        # Architecture Object. it contains stack of attention block
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type)
        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), 
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()
    
    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                constant_(p, 0.)

    def forward(self, q_data, target, pid_data=None):
        # Batch first
        qa_data = q_data + self.n_question * target
        q_embed_data = self.q_embed(q_data) # BS, seqlen, d_model # c_ct
        if self.separate_qa:
            # BS, seqlen, d_model #f_(ct, rt)
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # qa_data = (qa_data - q_data) // self.n_question # rt 
            # BS, seqlen, d_model # c_ct + g_rt = e_(ct, rt)
            qa_embed_data = self.qa_embed(target) + q_embed_data
        
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data) # d_ct
            pid_embed_data = self.difficult_param(pid_data) # uq
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data # uq * d_ct + c_ct
                
            qa_embed_diff_data = self.qa_embed_diff(
                target # 원래 qa_data
            ) # f_(ct, rt) or #h_rt
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data # uq * f_(ct, rt) + e_(ct, rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data + q_embed_diff_data) # + uq * (h_rt + d_ct)
            c_reg_loss = (pid_embed_data ** 2.).sum() * self.l2
        else:
            c_reg_loss = 0.

        # BS, seqlen, d_model
        # Pass to the decoder
        # output shae BS, seqlen, d_model or d_model // 2
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data) # 211x512

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)

        return preds, c_reg_loss


class Architecture(nn.Module):
    '''
        Args:
        n_block: number of stacked blocks in the attention
        d_model: dimension of attention input/output
        d_feature: dimension of input in each of the multi-head attention part.
        n_head: number of heads. n_heads * d_feature = d_model
    '''
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type
                ):
        super(Architecture, self).__init__()
        self.d_model = d_model
        self.model_type = model_type

        # iterate to given the number of blocks
        if model_type in {'akt'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads, 
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks * 2)
            ])
    
    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        # target shape bs, seqlen
        
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        for block in self.blocks_1: # encode qas
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        flag_first = True
        for block in self.blocks_2:
            if flag_first: # peek current question 
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False,  pdiff=pid_embed_data)
                flag_first = False
            else: # dont peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True,  pdiff=pid_embed_data)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    '''
        This is a Basic Block of Transformer paper. 
        It contains one Multi-head attention object. 
        Follwed by layer norm and position wise feed-forward net and dropout layer
    '''
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super(TransformerLayer, self).__init__()
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )
        
        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        '''
            Input

            Output:
                query: Input gets changed over the layer and returned.
        '''

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask
        ).astype('uint8')
        # src_mask = (torch.from_numpy(nopeek_mask) == 0).cuda()
        src_mask = (torch.from_numpy(nopeek_mask) == 0)
        if mask == 0: # If0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff
            )
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff
            )

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))
            ))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    '''
        It has projection layer for getting keys, queries and values.
        Followed by attention and a connected layer.
    '''
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        xavier_uniform_(self.gammas)

        self._reset_parameters()
    
    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)

        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)
    
    def forward(self, q, k, v, mask, zero_pad, pdiff=None):

        bs = q.size(0)
        
        # perform linear operation and split into h heads 
        k = self.k_linear(k)
        k = k.view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * s1 * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # calculate attention using function we will define next
        gammas = self.gammas
        
        scores = attention(q, k, v, self.d_k, 
                           mask, self.dropout, zero_pad, gammas, pdiff)
        
        # concatenate heads and put through final linear layer 
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)
        
        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """
        This is called by Multi-head attention object to find the values.
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k) # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # x1 = torch.arange(seqlen).expand(seqlen, -1).cuda()
    x1 = torch.arange(seqlen).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask.cuda() == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        # scores_ = scores_ * mask.float().cuda()
        scores_ = scores_ * mask.float().cuda()
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(
            scores_, dim=-1, keepdim=True 
        ) # bs, 8, s1, 1
        # position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).cuda() # 1, 1, seqlen, seqlen
        position_effect = torch.abs(x1-x2)[None, None, :, :].type(torch.FloatTensor).cuda() # 1, 1, seqlen, seqlen

        # bs, 8, s1, s1 positive distance
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.
        )
        dist_scores = dist_scores.sqrt().detach()
    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0) # 1, 8, 1, 1
    
    if pdiff == None:
        # Now after do exp(gamma * distance) and the clamp to 1e-5 to 1e5
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma).exp(), min=1e-5
        ), max=1e5)
    else: 
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1], pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma).exp(), min=1e-5
        ), max=1e5)
        
    scores = scores * total_effect

    scores.masked_fill_(mask.cuda() == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        # pad_zero = torch.zeros(bs, head, 1, seqlen).cuda()
        pad_zero = torch.zeros(bs, head, 1, seqlen)

        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


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
            y, preloss = model(q.long(), r.long(), q.long())

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            y = torch.masked_select(y, m)
            t = torch.masked_select(pid_seqs, m)

            opt.zero_grad()
            loss = binary_cross_entropy(y, t) + preloss.item() # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
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

                y, preloss = model(q.long(), r.long(), q.long())

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(pid_seqs, m).detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )
                bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
                acc = metrics.accuracy_score(t.numpy(), bin_y)
                loss = binary_cross_entropy(y, t) + preloss.item() # 원래는 preloss[0]으로 사용가능했음 (신버전 기준)
                
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

            y, preloss = model(q.long(), r.long(), q.long())

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            q = torch.masked_select(q, m).detach().cpu()
            y = torch.masked_select(pid_seqs, m).detach().cpu()
            t = torch.masked_select(r, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
            acc = metrics.accuracy_score(t.numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            loss = binary_cross_entropy(y, t) + preloss.item() # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

            print(f"[Test] number: {i}, AUC: {auc}, ACC: :{acc} Loss: {loss} ")

            # evaluation metrics
            aucs.append(auc)
            loss_mean.append(loss.detach().cpu().numpy())     
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
        loss_means.append(np.mean(loss_mean))


    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s