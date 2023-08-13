import os
import pickle

import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset
from models.utils import match_seq_len

# DATASET 변경이슈 (카카오 서버)
KAKAO_DATASET_DIR = "/app/input/dataset/dkt-dataset"
DATASET_DIR = f"{KAKAO_DATASET_DIR}/ASSIST2009/" # 원래는 datasets/ASSIST2009/

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
            with open(os.path.join(self.dataset_dir, Q_DIFF_PICKLE), "rb") as f:
                self.q2diff = pickle.load(f)
            with open(os.path.join(self.dataset_dir, P_ID_PICKLE), "rb") as f:
                self.pid_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, P_LIST_PICKLE), "rb") as f:
                self.pid_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, HINT_LIST_PICKLE), "rb") as f:
                self.hint_seqs = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.at_seqs, self.q_list, self.u_list, self.q2idx, \
                self.q2diff, self.pid_seqs, self.pid_list, self.hint_seqs = self.preprocess()
            
        
        def mapmax(data):
            return max(data, key=len)
        # 유저와 문제 갯수 저장
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_pid = self.pid_list.shape[0]

        self.wordlen = len(max(map(mapmax, self.at_seqs), key=len)) # 최대길이

        if seq_len:
            self.q_seqs, self.r_seqs, self.at_seqs, self.q2diff, self.pid_seqs, self.hint_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.at_seqs, self.q2diff, self.pid_seqs, self.hint_seqs, seq_len)

        self.len = len(self.q_seqs)
    
    def __getitem__(self, index) :
        return self.q_seqs[index], self.r_seqs[index], self.at_seqs[index], self.q2diff[index], self.pid_seqs[index], self.hint_seqs[index]
    
    def __len__(self):
        return self.len

    def preprocess(self):
        # 2번째 줄 dropna 내가 추가한 것이었으나 삭제 (# .dropna(subset=['answer_text'])\)
        df = pd.read_csv(self.dataset_path, encoding='ISO-8859-1').dropna(subset=["skill_name"])\
            .drop_duplicates(subset=["order_id", "skill_name"])\
            .sort_values(by=["order_id"])
        df['answer_text'] = df['answer_text'].fillna(' ')

        # 고유 유저와 고유 스킬리스트만 남김
        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values) 
        pid_list = np.unique(df["problem_id"].values)

        # map 형태로 스킬이름: index 자료 저장, 유저도 유저명: 인덱스로 저장
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        d2idx = {}
        p2idx = {pid: idx for idx, pid in enumerate(pid_list)}

        # 얼마나 썼는지 상관없이 힌트를 사용한 것으로 간주한다
        df.loc[df['hint_count'] >= 1, 'hint_count'] = 1
        df.loc[df['hint_count'] < 1, 'hint_count'] = 0

        q_seqs = []
        r_seqs = []
        at_seqs = []
        q2diff = []
        pid_seqs = []
        hint_seqs = []

        # 난이도 전처리, 미리 해당문제들의 정오 비율을 넣음
        for q in q_list:
            skills = df[df["skill_name"] == q]
            c_seq = skills["correct"].values
            d2idx[q] = c_seq.sum() / len(c_seq)

        for u in u_list:
            # 유저아이디를 순차적으로 돌면서 해당하는 유저 탐색
            df_u = df[df["user_id"] == u]

            # 유저의 스킬에 대한 해당 스킬의 인덱스와 정답 여부를 함께 시퀀스로 생성(여러 스킬의 경우에도 해당 인덱스 저장, 정답여부 저장) 
            # 스킬에 대한 인덱스 시퀀스와, 정답여부 시퀀스를 생성함
            q_seq = np.array([q2idx[q] for q in df_u["skill_name"]]) # 유저의 스킬에 대한 해당 스킬의 인덱스 리스트를 np.array로 형변환
            r_seq = df_u["correct"].values # 유저의 정답여부 저장
            at_seq = df_u['answer_text'].values
            pid_seq = np.array([p2idx[pid] for pid in df_u['problem_id']]) # 유저가 푼 문제에 대한 해당 문제의 인덱스 리스트들을 np.array로 형변환

            # 유저가 푼 문제들의 정오답 비율을 구함
            d_seq = np.array([d2idx[q] for q in df_u["skill_name"]])

            hint_seq = df_u['hint_count'].values

            # 해당 리스트들을 다시 리스트에 저장
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
            at_seqs.append(at_seq)
            q2diff.append(d_seq)
            pid_seqs.append(pid_seq)
            hint_seqs.append(hint_seq)

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
            pickle.dump(pid_seqs, f)
        with open(os.path.join(self.dataset_dir, P_LIST_PICKLE), "wb") as f:
            pickle.dump(pid_list, f)
        with open(os.path.join(self.dataset_dir, HINT_LIST_PICKLE), "wb") as f:
            pickle.dump(hint_seqs, f)

        return q_seqs, r_seqs, at_seqs, q_list, u_list, q2idx, q2diff, pid_seqs, pid_list, hint_seqs