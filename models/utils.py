import os
import numpy as np
import pickle
import torch 
from torch import nn
from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, CharTensor, LongTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor, CharTensor, LongTensor

from transformers import BertTokenizer, DistilBertTokenizer

from sklearn import metrics
from sklearn.metrics import classification_report
from torch.nn.functional import one_hot, binary_cross_entropy

import wandb

# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def match_seq_len(q_seqs, r_seqs, at_seqs, q2diff, pid_seqs, hint_seqs, seq_len, pad_val=-1):
    '''
        Args: 
            q_seqs: the question(KC) sequence with the size of \
                [batch_size, some_sequence_length]
            r_seqs: the response sequence with the size of \
                [batch_size, some_sequence_length]
            at_seqs: the answer text sequence with the size of \
                [batch_size, some_sequence_length]
            
            Note that the "some_sequence_length" is not uniform over \
                the whole batch of q_seqs and r_seqs
            
            seq_len: the sequence length to match the q_seqs, r_seqs \
                to same length
            pad_val: the padding value for the sequence with the length \
                longer than seq_len

        Returns:
            proc_q_seqs: the processed q_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_r_seqs: the processed r_seqs with the size of \
                [batch_size, seq_len + 1]
            proc_at_seqs: the processed at_seqs with the size of \
                [batch_size, seq_len + 1]
    '''

    proc_q_seqs = []
    proc_r_seqs = []
    proc_at_seqs = []
    proc_q2diff = []
    proc_pid_seqs = []
    proc_hint_seqs = []

    # seq_len은 q_seqs와 r_seqs를 같은 길이로 매치하는 시퀀스 길이를 의미함.
    # q_seq는 유저의 스킬에 대한 인덱스 리스트를 갖는 리스트임.
    # 주어진 q, r시퀀스들을 seq_len 만큼 자르는 것이라고 보면 됨
    for q_seq, r_seq, at_seq, q2d, pid_seq, hint_seq in zip(q_seqs, r_seqs, at_seqs, q2diff, pid_seqs, hint_seqs):
        i = 0
        while i + seq_len + 1 < len(q_seq): # i + seq_len + 1 이 주어진 문제 집합보다 길이가 작을 때, e.g.) 0 + 100 + 1 < 128
            proc_q_seqs.append(q_seq[i:i + seq_len + 1]) # i부터 i + seq_len + 1 범위의 elements를 퀘스천 시퀀스에 넣음 e.g.) 0부터 0 + 100 + 1 원소의 배열 시퀀스를 proc_q에 할당함
            proc_r_seqs.append(r_seq[i:i + seq_len + 1]) # 위와 동일. e.g.) 0부터 0 + 100 + 1 원소 배열 시퀀스를 proc_r에 할당함
            proc_at_seqs.append(at_seq[i:i + seq_len + 1])
            proc_q2diff.append(q2d[i:i + seq_len + 1])
            proc_pid_seqs.append(pid_seq[i:i + seq_len + 1])
            proc_hint_seqs.append(hint_seq[i:i + seq_len + 1])

            i += seq_len + 1 # i에 seq_len + 1을 더하여 len(q_seq)보다 크게 만듬

        # seq_len 만큼 자른 sequence들을 padding값이 들어가 있는 합쳐서 넣음
        # 자른 건 길이 모자라니깐 padding 값으로 대체해서 넣음, 아닐 경우 원래 시퀀스에 패딩값이 들어있는 배열 붙여넣음
        proc_q_seqs.append(
            np.concatenate([
                q_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq))) # padding value가 들어있는 배열의 원소를 * 갯수만큼 생성함 (여기선 0, seq_len q_seq가 128이라 가정하면 129 - 128, 즉 1개만 만듬)
            ]) 
        )
        proc_r_seqs.append(
            np.concatenate([
                r_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            ])
        )
        proc_at_seqs.append(
            np.concatenate([
                at_seq[i:],
                np.array([' '] * (i + seq_len + 1 - len(at_seq)))
            ])
        )
        proc_q2diff.append(
            np.concatenate([
                q2d[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
            ])
        )
        proc_pid_seqs.append(
            np.concatenate([
                pid_seq[i:],
                np.array([pad_val] * (i + seq_len + 1 - len(pid_seq)))
            ])
        )
        proc_hint_seqs.append(
            np.concatenate(
                [
                    hint_seq[i:],
                    np.array([pad_val] * (i + seq_len + 1 - len(q_seq)))
                ]
            )
        )
        # 마지막 1개의 원소들은 패딩해서 넣게 됨

    return proc_q_seqs, proc_r_seqs, proc_at_seqs, proc_q2diff, proc_pid_seqs, proc_hint_seqs

def collate_ednet(batch, pad_val=-1):
    '''
        아래서는 at : answer_text 였듯이
        여기서는 t: 가 optional answer임.
    '''
    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    t_seqs = []
    tshft_seqs = []
    
    for q_seq, r_seq, t_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1]))
        r_seqs.append(FloatTensor(r_seq[:-1]))
        t_seqs.append(t_seq[:-1])
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        tshft_seqs.append(FloatTensor(t_seq[1:]))
    
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )
    t_seqs = pad_sequence(
        t_seqs, batch_first=True, padding_value=pad_val
    )
    tshft_seqs = pad_sequence(
        tshft_seqs, batch_first=True, padding_value=pad_val
    )
    
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    q_seqs, r_seqs, qshft_seqs, rshft_seqs, t_seqs, tshft_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs, t_seqs * mask_seqs, tshft_seqs * mask_seqs
        
    bert_details = []
    
    for answer_text in t_seqs:
        # text = ' '.join(answer_text)
        text = list(map(str), answer_text)
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, add_special_tokens=False, truncation=True, return_token_type_ids=False
        )
        bert_details.append(encoded_bert_sent)
    
    bert_sentences = LongTensor([text["input_ids"] for text in bert_details])
    # bert_sentence_types = LongTensor([text["token_type_ids"] for text in bert_details])
    bert_sentence_att_mask = LongTensor([text["attention_mask"] for text in bert_details])
    
    
    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, [], bert_sentence_att_mask, [], [], [], []


def collate_fn(batch, pad_val=-1):
    '''
    This function for torch.utils.data.DataLoader

    Returns:
        q_seqs: the question(KC) sequences with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        r_seqs: the response sequences with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        qshft_seqs: the question(KC) sequences which were shifted \
            one step to the right with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        rshft_seqs: the response sequences which were shifted \
            one step to the right with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
        mask_seqs: the mask sequences indicating where \
            the padded entry is with the size of \
            [batch_size, maximum_sequence_length_in_the_batch]
    '''

    q_seqs = []
    r_seqs = []
    qshft_seqs = []
    rshft_seqs = []
    at_seqs = []
    atshft_seqs = []
    q2diff_seqs = []
    pid_seqs = []
    pidshft_seqs = []
    hint_seqs = []

    # q_seq와 r_seq는 마지막 전까지만 가져옴 (마지막은 padding value)
    # q_shft와 rshft는 처음 값 이후 가져옴 (우측 시프트 값이므로..)
    for q_seq, r_seq, at_seq, q2diff, pid_seq, hint_seq in batch:
        q_seqs.append(FloatTensor(q_seq[:-1])) 
        r_seqs.append(FloatTensor(r_seq[:-1]))
        at_seqs.append(at_seq[:-1])
        atshft_seqs.append(at_seq[1:])
        qshft_seqs.append(FloatTensor(q_seq[1:]))
        rshft_seqs.append(FloatTensor(r_seq[1:]))
        q2diff_seqs.append(FloatTensor(q2diff[:-1]))
        pid_seqs.append(FloatTensor(pid_seq[:-1]))
        pidshft_seqs.append(FloatTensor(pid_seq[1:]))
        hint_seqs.append(FloatTensor(hint_seq[:-1]))

    # pad_sequence, 첫번째 인자는 sequence, 두번째는 batch_size가 첫 번째로 인자로 오게 하는 것이고, 3번째 인자의 경우 padding된 요소의 값
    # 시퀀스 내 가장 길이가 긴 시퀀스를 기준으로 padding이 됨, 길이가 안맞는 부분은 늘려서 padding_value 값으로 채워줌
    q_seqs = pad_sequence(
        q_seqs, batch_first=True, padding_value=pad_val
    )
    r_seqs = pad_sequence(
        r_seqs, batch_first=True, padding_value=pad_val
    )
    q2diff_seqs = pad_sequence(
        q2diff_seqs, batch_first=True, padding_value=pad_val
    )
    qshft_seqs = pad_sequence(
        qshft_seqs, batch_first=True, padding_value=pad_val
    )
    rshft_seqs = pad_sequence(
        rshft_seqs, batch_first=True, padding_value=pad_val
    )
    pid_seqs = pad_sequence(
        pid_seqs, batch_first=True, padding_value=pad_val
    )
    pidshft_seqs = pad_sequence(
        pidshft_seqs, batch_first=True, padding_value=pad_val
    )
    hint_seqs = pad_sequence(
        hint_seqs, batch_first=True, padding_value=pad_val
    )

    # 마스킹 시퀀스 생성 
    # 일반 question 시퀀스: 패딩 밸류와 다른 값들은 모두 1로 처리, 패딩 처리된 값들은 0으로 처리.
    # 일반 question padding 시퀀스: 한 칸 옆으로 시프팅 된 시퀀스 값들이 패딩 값과 다를 경우 1로 처리, 패딩 처리 된 값들은 0으로 처리.
    # 마스킹 시퀀스: 패딩 처리 된 시퀀스 밸류들은 모두 0, 두 값 모두 패딩처리 되지 않았을 경우 1로 처리. (원본 시퀀스와 shift 시퀀스 모두의 값)
    # 예를 들어, 현재 값과 다음 값이 패딩 값이 아닐 경우 1, 현재 값과 다음 값 둘 중 하나라도 패딩일 경우 0으로 처리함.
    mask_seqs = (q_seqs != pad_val) * (qshft_seqs != pad_val)

    # 원본 값의 다음 값이(shift value) 패딩이기만 해도 마스킹 시퀀스에 의해 값이 0로 변함. 아닐경우 원본 시퀀스 데이터를 가짐.
    q_seqs, r_seqs, qshft_seqs, rshft_seqs, q2diff_seqs, pid_seqs, pidshft_seqs, hint_seqs = \
        q_seqs * mask_seqs, r_seqs * mask_seqs, qshft_seqs * mask_seqs, \
        rshft_seqs * mask_seqs, q2diff_seqs * mask_seqs, pid_seqs * mask_seqs, \
        pidshft_seqs * mask_seqs, hint_seqs * mask_seqs
    

    # Word2vec

    # BERT preprocessing
    bert_details = []

    # def mapmax(data):
    #     return max(data, key=len)

    # 2차원에서 가장 긴 문장 추출
    # SENT_LEN = len(max(map(mapmax, at_seqs), key=len))

    for answer_text in at_seqs:
        # text = ' '.join(map(str, answer_text))
        text = list(map(str), answer_text)

        # print(f"============= text: {text} ================")
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, add_special_tokens=True, padding='max_length', truncation=True, return_token_type_ids=True
        )
        bert_details.append(encoded_bert_sent)
    
    # 정답지 추가
    # proc_atshft_seqs = []
    # # SENT_LEN = q_seqs.size(0)
    # for answer_text in atshft_seqs:
    #     text = " ".join(map(str, answer_text))
    #     encoded_bert_sent = bert_tokenizer.encode_plus(
    #         text, add_special_tokens=True, padding='max_length', truncation=True
    #     )
    #     proc_atshft_seqs.append(encoded_bert_sent)

    bert_sentences = LongTensor([text["input_ids"] for text in bert_details])
    bert_sentence_types = LongTensor([text["token_type_ids"] for text in bert_details])
    bert_sentence_att_mask = LongTensor([text["attention_mask"] for text in bert_details])
    # proc_atshft_sentences = LongTensor([text["input_ids"] for text in proc_atshft_seqs])

    return q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, q2diff_seqs, pid_seqs, pidshft_seqs, hint_seqs

    
def equalized_odd(y_pred, y_true, sensitive, lambda_s=0.3):
    """
    
    """

    # unique한 속성만 남김
    sensitive_group = torch.unique(sensitive)

    fpr, tpr = [], []

    for group in sensitive_group:
        group_mask = (sensitive == group) # 마스킹

        true_positives = (y_pred[group_mask] >= 0.5).float().sum() # 모델이 양성으로 예측한 경우, 실제 값도 양성
        false_positives = (y_pred[group_mask] < 0.5).float().sum() # 모델이 음성으로 양성으로 예측한 경우, 실제 값은 음성
        true_negatives = (y_pred[~group_mask] < 0.5).float().sum() # 모델이 음성으로 예측한 경우, 실제 값은 음성
        false_negatives = (y_pred[~group_mask] >= 0.5).float().sum() # 모델이 음성으로 예측한 경우, 실제 값은 양성

        fpr_group = false_positives / (false_positives + true_negatives)
        tpr_group = true_positives / (true_positives + false_negatives)

        fpr.append(fpr_group)
        tpr.append(tpr_group)

    fpr = torch.tensor(fpr)
    tpr = torch.tensor(tpr)

    fpr_diff = torch.abs(torch.max(fpr) - torch.min(fpr))
    tpr_diff = torch.abs(torch.max(tpr) - torch.min(tpr))

    eq_odd = fpr_diff + tpr_diff
    regularization = lambda_s * eq_odd

    return regularization, eq_odd

def calculate_dis_impact(y_pred, y_true, sensitive):

    _ = None

    sensitive_group = torch.unique(sensitive)

    predictive_parity_scores = []

    for group in sensitive_group:
        group_indices = sensitive == group
        group_y_true = y_true[group_indices]
        group_y_pred = y_pred[group_indices]

        group_positive_prob = torch.mean(group_y_pred[group_y_true == 1])

        group_diff = torch.abs(group_positive_prob - torch.mean(y_pred))

        predictive_parity_scores.append(group_diff)

    predictive_parity_score = torch.mean(torch.tensor(predictive_parity_scores))

    return _, predictive_parity_score

def equal_opportunity(y_pred, y_true, sensitive):
    """
    
    """

    # unique한 속성만 남김
    sensitive_group = torch.unique(sensitive)
    tpr_diff = 0

    tpr_group = []
    for group in sensitive_group:
        group_mask = (sensitive == group) # 마스킹
        true_positives = (y_pred[group_mask] > 0).float().sum()  
        false_negatives = (y_pred[~group_mask] > 0).float().sum()   

        tpr_group.append(true_positives / (true_positives + false_negatives))

    length = len(tpr_group)
    for i in range(0, length):
        now_sensit = tpr_group[i]

        if i == 0:
            next_sensit = tpr_group[i + 1]
            tpr_diff =  torch.abs(torch.tensor(now_sensit) - torch.tensor(next_sensit))
        elif length == (i + 1):
            break
        else: 
            tpr_diff = torch.abs(tpr_diff - torch.tensor(next_sensit))
    tpr_group = torch.tensor(tpr_group)
    eq_opp = tpr_group.mean()
    regularization = tpr_diff

    return regularization, eq_opp

def cal_acc_class(q_seqs, y_true, y_pred):
    # convert to numpy, but y_pred is list
    q_seqs_np = q_seqs.numpy()
    y_true_np = y_true.numpy()
    y_pred_np = y_pred
    
    # intialize a dictionary to store the accuracy of each question.
    question_cnts = {}
    question_corrects = {}
    accs = {}
    
    for idx, (question, correct, pred_correct) in enumerate(zip(q_seqs_np, y_true_np, y_pred_np)):
        
        question_id = question.item()        
        correctness = 1 if pred_correct == correct.item() else 0
                
        # caculate the count and correctness of the question
        if(question_id in question_cnts):
            question_cnts[question_id] += 1
            question_corrects[question_id] += correctness
        else: 
            question_cnts[question_id] = 1
            question_corrects[question_id] = correctness
    
    # caculate the accuracy of the question
    for ((cnt_k, cnt_v), (cr_k, cr_v)) in zip(question_cnts.items(), question_corrects.items()):
        
        # 해당 문제 번호에 accuracy 저장 (맞춘 수 / 총 문제 수)
        accs[cnt_k] = cr_v / cnt_v

    
    return accs, question_cnts


### related to train parts
def reset_weight(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()
            
def early_stopping(best_loss, loss, patience_check):
    # 로스가 개선되지 않았을 경우
    if loss > best_loss:
        patience_check += 1
        
        # if patience_check >= patience_limit:
        #     return patience_check # break 걸어줘서 stopping
    else:
        best_loss = loss
        patience_check = 0
        # patience_check 초기화 
    return patience_check
    
def save_auc(model, max_auc, auc, hyp_dict, ckpt_path, use_wandb):
    if auc > max_auc : 
        torch.save(
            model.state_dict(),
            os.path.join(
                ckpt_path, "model.ckpt"
            )
        )
        max_auc = auc
        if(use_wandb == True):
            with open(os.path.join(ckpt_path, f"best_val_auc.pkl"), "wb") as f:
                best_pef = hyp_dict
                # e.g
                # {"seed": wandb.config.seed, \
                #         "dropout": wandb.config.dropout, \
                #         "lr": wandb.config.learning_rate, \
                #         'dim_s': {'values': [20, 50]}, \
                #         'size_m': {'values': [20, 50]}
                #         # "emb_size": wandb.config.emb_size, \
                #         # "hidden_size": wandb.config.hidden_size \
                #         }
                pickle.dump(best_pef, f)
    return max_auc 

def save_auc_bert(model, max_auc, auc, hyp_dict, ckpt_path, use_wandb):
    if auc > max_auc : 
        torch.save(
            model.state_dict(),
            os.path.join(
                ckpt_path, "model.ckpt"
            )
        )
        max_auc = auc
        
        bert_tokenizer.save_vocabulary(os.path.join(ckpt_path, "tokenizer.ckpt"))
        
        if(use_wandb == True):
            with open(os.path.join(ckpt_path, f"best_val_auc.pkl"), "wb") as f:
                best_pef = hyp_dict
                # e.g
                # {"seed": wandb.config.seed, \
                #         "dropout": wandb.config.dropout, \
                #         "lr": wandb.config.learning_rate, \
                #         'dim_s': {'values': [20, 50]}, \
                #         'size_m': {'values': [20, 50]}
                #         # "emb_size": wandb.config.emb_size, \
                #         # "hidden_size": wandb.config.hidden_size \
                #         }
                pickle.dump(best_pef, f)
    return max_auc 

            
def log_auc(use_wandb, log_dict):
    if(use_wandb != False):
        wandb.log(log_dict)        
        # e.g{
        #     "epoch": epoch,
        #     "train_auc": auc_mean, 
        #     "train_acc": acc_mean,
        #     "train_loss": loss_mean
        # }

def common_train(model, opt, q, r, m):
    inpt_q = q.long() 
    inpt_r = r.long()
    
    y = model(inpt_q, inpt_r)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(r, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    loss.backward()
    opt.step()
    
    return y, t, loss

def dkt_train(model, opt, q, r, qshft_seqs, rshft_seqs, num_q, m):
    inpt_q = q.long()
    inpt_r = r.long()
    next_q = qshft_seqs.long()
    
    y = model(inpt_q, inpt_r)
    y = (y * one_hot(next_q, num_q)).sum(-1)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(rshft_seqs, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    loss.backward()
    opt.step()
    
    return y, t, loss

def dkvmn_train(model, opt, q, r, m):
    inpt_q = q.long() 
    inpt_r = r.long()
    
    y, Mv = model(inpt_q, inpt_r)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(r, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    loss.backward()
    opt.step()
    
    return y, t, loss

def dkvmn_bert_train(model, opt, q, r, at_s, at_t, at_m, m):
    inpt_q = q.long()
    inpt_r = r.long()
    
    y, Mv = model(inpt_q, inpt_r, at_s, at_t, at_m)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(r, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    loss.backward()
    opt.step()
    
    return y, t, loss

def sakt_train(model, opt, q, r, qshft_seqs, rshft_seqs, m):
    inpt_q = q.long()
    inpt_r = r.long()
    next_q = qshft_seqs.long()
    next_r = rshft_seqs.long()
    
    # 현재까지의 입력을 받은 뒤 다음 문제 예측
    y, _ = model(inpt_q, inpt_r, next_q)

    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(rshft_seqs, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t)
    loss.backward()
    opt.step()
            
    return y, t, loss

def akt_train(model, opt, q, r, pid, m):
    inpt_q = q.long()
    inpt_r = r.long()
    inpt_pid = pid.long()
    
    y, preloss = model(inpt_q, inpt_r, inpt_pid)

    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(r, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t) + preloss.item() # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    loss.backward()
    opt.step()
    
    return y, t, loss

def bert_train(model, opt, q, r, m, at_s, at_t, at_m):
    inpt_q = q.long()
    inpt_r = r.long()
    
    y = model(inpt_q, inpt_r, at_s, at_t, at_m)
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    y = torch.masked_select(y, m)
    t = torch.masked_select(r, m)
    
    opt.zero_grad()
    loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    loss.backward()
    opt.step()
    
    return y, t, loss 

def common_test(model, q, r, m):
    inpt_q = q.long()
    inpt_r = r.long()
    
    y = model(inpt_q, inpt_r)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    q = torch.masked_select(q, m).detach().cpu()
    y = torch.masked_select(y, m).detach().cpu()
    t = torch.masked_select(r, m).detach().cpu()
    
    loss = binary_cross_entropy(y, t)
    
    return q, y, t, loss

def dkt_test(model, q, r, qshft_seqs, rshft_seqs, num_q, m):
    inpt_q = q.long()
    inpt_r = r.long()
    
    y = model(inpt_q, inpt_r)
    y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    q = torch.masked_select(q, m).detach().cpu()
    y = torch.masked_select(y, m).detach().cpu()
    t = torch.masked_select(rshft_seqs, m).detach().cpu()
    
    loss = binary_cross_entropy(y, t)
    return q, y, t, loss

def dkvmn_test(model, q, r, m):
    inpt_q = q.long()
    inpt_r = r.long()
    
    y, Mv = model(inpt_q, inpt_r)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    q = torch.masked_select(q, m).detach().cpu()
    y = torch.masked_select(y, m).detach().cpu()
    t = torch.masked_select(r, m).detach().cpu()
    
    loss = binary_cross_entropy(y, t)
    return q, y, t, loss, Mv

def dkvmn_bert_test(model, q, r, at_s, at_t, at_m, m):
    inpt_q = q.long()
    inpt_r = r.long()
    
    y, Mv = model(inpt_q, inpt_r, at_s, at_t, at_m)
    
    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    q = torch.masked_select(q, m).detach().cpu()
    y = torch.masked_select(y, m).detach().cpu()
    t = torch.masked_select(r, m).detach().cpu()
    
    loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    
    return q, y, t, loss, Mv

def sakt_test(model, q, r, qshft_seqs, rshft_seqs, m):
    inpt_q = q.long()
    inpt_r = r.long()
    next_q = qshft_seqs.long()
    next_r = rshft_seqs
    
    # 현재까지의 입력을 받은 뒤 다음 문제 예측
    y, Aw = model(inpt_q, inpt_r, next_q)

    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    q = torch.masked_select(q, m).detach().cpu()
    y = torch.masked_select(y, m).detach().cpu()
    t = torch.masked_select(next_r, m).detach().cpu()
    
    loss = binary_cross_entropy(y, t)

    return q, y, t, loss, Aw

def akt_test(model, q, r, pid, m):
    inpt_q = q.long()
    inpt_r = r.long()
    inpt_pid = pid.long()
    
    y, preloss = model(inpt_q, inpt_r, inpt_pid)

    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
    q = torch.masked_select(q, m).detach().cpu()
    y = torch.masked_select(y, m).detach().cpu()
    t = torch.masked_select(r, m).detach().cpu()
    
    loss = binary_cross_entropy(y, t) + preloss.item() # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
    
    return q, y, t, loss

def common_append(y, t, loss, loss_mean, auc_mean, acc_mean):
    
    loss_mean.append(
        loss.detach().cpu().numpy()
    )
    auc_mean.append(metrics.roc_auc_score(
        y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy()
    ))
    bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
    acc_mean.append(metrics.accuracy_score(t.detach().cpu().numpy(), bin_y))
    return bin_y
    
def val_append(t, bin_y, precision_mean, recall_mean, f1_mean):
    target = t.numpy()
    
    precision = metrics.precision_score(target, bin_y, average='binary')
    recall = metrics.recall_score(target, bin_y, average='binary')
    f1 = metrics.f1_score(target, bin_y, average='binary')
    
    precision_mean.append(precision)
    recall_mean.append(recall)
    f1_mean.append(f1)

def mean_eval(loss_mean, auc_mean, acc_mean):
    
    lm = np.mean(loss_mean)
    aum = np.mean(auc_mean)
    acm = np.mean(acc_mean)
    
    return lm, aum, acm

def mean_eval_ext(precision_mean, recall_mean, f1_mean):
    
    pm = np.mean(precision_mean)
    rm = np.mean(recall_mean)
    f1m = np.mean(f1_mean)
    
    return pm, rm, f1m