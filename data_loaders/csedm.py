import torch
from torchvision import transforms
from models.utils import match_seq_len
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import pickle

KAKAO_DATASET_DIR = "/app/input/dataset/dkt-dataset"
DATASET_DIR = f"{KAKAO_DATASET_DIR}/CSEDM/" 
# KAKAO 때문에 추가
SAVE_DIR = "/app/outputs/"
# DATASET_DIR = "datasets/CSEDM/"
Q_SEQ_PICKLE = "q_seqs.pkl"
R_SEQ_PICKLE = "r_seqs.pkl"
U_SEQ_PICKLE = "u_seqs.pkl"
T_SEQ_PICKLE = "t_seqs.pkl"
# U_LIST_PICKLE = "u_list.pkl"
# Q_IDX_PICKLE = "q2idx.pkl"
# Q_DIFF_PICKLE = 'q2diff.pkl'
# P_ID_PICKLE = 'pid.pkl'
# P_LIST_PICKLE = "p_list.pkl"
# HINT_LIST_PICKLE = "hint_use.pkl"

Q_SEQ_PICKLE = "q_seqs.pkl"
R_SEQ_PICKLE = "r_seqs.pkl"
AT_SEQ_PICKLE = "at_seqs.pkl"
Q_LIST_PICKLE = "q_list.pkl"
U_LIST_PICKLE = "u_list.pkl"
Q_IDX_PICKLE = "q2idx.pkl"
Q_DIFF_PICKLE = 'q2diff.pkl'
P_ID_PICKLE = 'pid.pkl'
P_LIST_PICKLE = "p_list.pkl"
HINT_LIST_PICKLE = "hint_use.pkl"

class CSEDM(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.save_dir = SAVE_DIR
        self.dataset_path = os.path.join(
            self.dataset_dir, "ALL_CSEDM.csv"
        )

        # 미리 피클에 담겨진 파일들 로딩
        if os.path.exists(os.path.join(self.dataset_dir, Q_SEQ_PICKLE)) & os.path.exists(os.path.join(self.dataset_dir, 'ALL_CSEDM.csv')):
            with open(os.path.join(self.dataset_dir, Q_SEQ_PICKLE), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, R_SEQ_PICKLE), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, Q_LIST_PICKLE), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, U_LIST_PICKLE), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, AT_SEQ_PICKLE), "rb") as f:
                self.at_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, Q_DIFF_PICKLE), "rb") as f:
                self.q2diff = pickle.load(f)
            with open(os.path.join(self.dataset_dir, P_ID_PICKLE), "rb") as f:
                self.pid_list = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.at_seqs, self.q_list, \
            self.u_list, self.q2idx, self.q2diff, self.pid_list = self.preprocess()

        # 유저와 문제 갯수 저장
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_pid = self.pid_list.shape[0]
        
        
        if seq_len:
            self.q_seqs, self.r_seqs, self.at_seqs, self.q2diff, _, _ = \
                match_seq_len(self.q_seqs, self.r_seqs, self.at_seqs, self.q2diff, self.q_seqs, self.q_seqs, seq_len) # 3개는 더미임

        self.len = len(self.q_seqs)
        
    def __getitem__(self, index) :
        # self.pid_seqs[index], self.hint_seqs[index]
        return self.q_seqs[index], self.r_seqs[index], self.at_seqs[index], self.q2diff[index], self.q_seqs[index], self.q_seqs[index] # 여기도 2개 더미임
    
    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, encoding='utf-8-sig')
        df['Label'] = df['Label'].replace({True: 1, False: 0})        
        
        # 고유 유저와 고유 스킬리스트만 남김
        u_list = np.unique(df["SubjectID"].values)
        q_list = np.unique(df["ProblemID"].values) 
        pid_list = np.unique(df["AssignmentID"].values)

        # map 형태로 스킬이름: index 자료 저장, 유저도 유저명: 인덱스로 저장
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        d2idx = {}

        q_seqs = []
        r_seqs = []
        at_seqs = []
        q2diff = []

        # 난이도 전처리, 미리 해당문제들의 정오 비율을 넣음
        for q in q_list:
            skills = df[df["ProblemID"] == q]
            c_seq = skills["Label"].values
            d2idx[q] = c_seq.sum() / len(c_seq)

        for u in u_list:
            # 유저아이디를 순차적으로 돌면서 해당하는 유저 탐색
            df_u = df[df["SubjectID"] == u]

            # 유저의 스킬에 대한 해당 스킬의 인덱스와 정답 여부를 함께 시퀀스로 생성(여러 스킬의 경우에도 해당 인덱스 저장, 정답여부 저장) 
            # 스킬에 대한 인덱스 시퀀스와, 정답여부 시퀀스를 생성함
            q_seq = np.array([q2idx[q] for q in df_u["ProblemID"]]) # 유저의 스킬에 대한 해당 스킬의 인덱스 리스트를 np.array로 형변환
            r_seq = df_u["Label"].values # 유저의 정답여부 저장
            at_seq = df_u['Code'].values

            # 유저가 푼 문제들의 정오답 비율을 구함
            d_seq = np.array([d2idx[q] for q in df_u["ProblemID"]])

            # 해당 리스트들을 다시 리스트에 저장
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            at_seqs.append(at_seq)
            q2diff.append(d_seq)

        with open(os.path.join(self.dataset_dir, Q_SEQ_PICKLE), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, R_SEQ_PICKLE), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, AT_SEQ_PICKLE), "wb") as f:
            pickle.dump(at_seqs, f)
        with open(os.path.join(self.dataset_dir, Q_LIST_PICKLE), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, U_LIST_PICKLE), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, Q_IDX_PICKLE), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, Q_DIFF_PICKLE), "wb") as f:
            pickle.dump(q2diff, f)
        with open(os.path.join(self.dataset_dir, P_ID_PICKLE), "wb") as f:
            pickle.dump(pid_list, f)

        return q_seqs, r_seqs, at_seqs, q_list, u_list, q2idx, q2diff, pid_list