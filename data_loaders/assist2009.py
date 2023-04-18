import os
import pickle

import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset
from models.utils import match_seq_len

DATASET_DIR = "datasets/ASSIST2009/"
Q_SEQ_PICKLE = "q_seqs.pkl"
R_SEQ_PICKLE = "r_seqs.pkl"
AT_SEQ_PICKLE = "at_seqs.pkl"
Q_LIST_PICKLE = "q_list.pkl"
U_LIST_PICKLE = "u_list.pkl"
Q_IDX_PICKLE = "q2idx.pkl"
U_IDX_PICKLE = 'u2idx.pkl'

class ASSIST2009(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_dir = dataset_dir
        self.dataset_path = os.path.join(
            self.dataset_dir, "skill_builder_data.csv"
        )

        # 미리 피클에 담겨진 파일들 로딩
        if os.path.exists(os.path.join(self.dataset_dir, Q_SEQ_PICKLE)):
            with open(os.path.join(self.dataset_dir, Q_SEQ_PICKLE), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, R_SEQ_PICKLE), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, AT_SEQ_PICKLE), "rb") as f:
                self.at_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, Q_LIST_PICKLE), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, U_LIST_PICKLE), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, Q_IDX_PICKLE), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, U_IDX_PICKLE), "rb") as f:
                self.u2idx = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.at_seqs, self.q_list, self.u_list, self.q2idx, \
                self.u2idx = self.preprocess()
        
        # 유저와 문제 갯수 저장
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if seq_len:
            self.q_seqs, self.r_seqs, self.at_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.at_seqs, seq_len)

        self.len = len(self.q_seqs)
    
    def __getitem__(self, index) :
        return self.q_seqs[index], self.r_seqs[index], self.at_seqs[index]
    
    def __len__(self):
        return self.len

    def preprocess(self):

        df = pd.read_csv(self.dataset_path, encoding='ISO-8859-1').dropna(subset=["skill_name"])\
            .drop_duplicates(subset=["order_id", "skill_name"])\
            .sort_values(by=["order_id"])
        
        # 고유 유저와 고유 스킬리스트만 남김
        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values) 

        # map 형태로 스킬이름: index 자료 저장, 유저도 유저명: 인덱스로 저장
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        q_seqs = []
        r_seqs = []
        at_seqs = []

        for u in u_list:
            # 유저아이디를 순차적으로 돌면서 해당하는 유저 탐색
            df_u = df[df["user_id"] == u]

            # 유저의 스킬에 대한 해당 스킬의 인덱스와 정답 여부를 함께 시퀀스로 생성(여러 스킬의 경우에도 해당 인덱스 저장, 정답여부 저장) 
            # 스킬에 대한 인덱스 시퀀스와, 정답여부 시퀀스를 생성함
            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]]) # 유저의 스킬에 대한 해당 스킬의 인덱스 리스트를 np.array로 형변환
            r_seq = df_u["correct"].values # 유저의 정답여부 저장
            at_seq = df_u["answer_text"].values

            # 해당 리스트들을 다시 리스트에 저장
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            at_seqs.append(at_seq)

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
        with open(os.path.join(self.dataset_dir, U_IDX_PICKLE), "wb") as f:
            pickle.dump(u2idx, f)

        return q_seqs, r_seqs, at_seqs, q_list, u_list, q2idx, u2idx