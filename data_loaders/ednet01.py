import torch
from torchvision import transforms
# from constant import MAX_STEP, NUM_OF_QUESTIONS  
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import pickle


DATASET_DIR = "datasets/EDNET01/"
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

# 커스텀 데이터셋은 클래스 상속받아 사용
class EdNet01(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR):
        super().__init__()

        self.dataset_dir = dataset_dir

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
                
        self.dataset_path = os.path.join(
            self.dataset_dir, "Ednet01.csv"
        )

        # 유저와 문제 갯수 저장
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        self.num_pid = self.pid_list.shape[0]

        self.len = len(self.q_seqs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        print(self.q_seqs[index], self.r_seqs[index], self.t_seqs[index])
        return self.q_seqs[index], self.r_seqs[index], self.t_seqs[index]


    def preprocess(self):
        path = self.pickle_path

        u_seqs = []
        qid_seqs = []
        r_seqs = []
        t_seqs = []

        # 새로운 csv 파일 저장을 위한 딕셔너리 생성
        concate_csv = {}
        
        # 파일 전체적으로 읽기
        print(os.getcwd())
        file_list = os.listdir("./data_loaders/dataset/KT1_train") # 파일리스트 로드
        # 정답 비교를 위한 파일 추가
        q_df = pd.read_csv(f"./data_loaders/dataset/KT1_train/questions.csv")
        q_df['question_id'] = q_df['question_id'].astype(str)
        
        # 파일리스트 로드 후 불러오기 (로딩 적용)
        for file in tqdm(file_list, leave=False):
            u_id = f'{file}'
            u_df = pd.read_csv(f"/data_loaders/dataset/KT1_train/{file}")
            # 가끔 안되는게 있어서 타입 변경
            u_df['question_id'] = u_df['question_id'].astype(str)
            u_df['correct'] = u_df['correct'].where(q_df['question_id'] == u_df['question_id'] and u_df["user_answer"] == q_df['correct_answer'], 1)
            u_df['correct'] = u_df['correct'].where(u_df['correct'] != 1, 0)
            
            # 유저 아이디 시퀀스에 넣기
            u_seqs.append(u_id)

            qid_seqs.append([u_df['question_id'].values])
            t_seqs.append([u_df['correct'].values])
            r_seqs.append([u_df['user_answer'].values])
            
            concate_csv[file] = u_df
            
        # 피클에 파일 저장
        with open(f"{path}/q_seqs.pkl", "wb") as f:
            pickle.dump(qid_seqs, f)
        with open(f"{path}/r_seqs.pkl", "wb") as f:
            pickle.dump(r_seqs, f)
        with open(f"{path}/u_seqs.pkl", "wb") as f:
            pickle.dump(u_seqs, f)
        with open(f"{path}/t_seqs.pkl", "wb") as f:
            pickle.dump(t_seqs, f)

        # 새로 만든 파일 저장
        new_file = pd.concat(concate_csv.values(), ignore_index=True)
        new_file.to_csv('Ednet01.csv', index=False)
        
        return qid_seqs, r_seqs, u_seqs, t_seqs