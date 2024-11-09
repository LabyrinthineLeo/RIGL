# -*- coding: utf-8 -*-
import os, datetime, json
import random as python_random
from torch import nn
from torch.nn import Module, Linear, Dropout,Sequential, ReLU
import copy
import pandas as pd
import numpy as np
import torch

device = "cpu" if not torch.cuda.is_available() else "cuda"


def set_seed(seed):
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as e:
        print("Set seed failed, details are ", e)
        pass

    np.random.seed(seed)
    python_random.seed(seed)

    # cuda env
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def get_now_time():
    now = datetime.datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return dt_string


def debug_print(text, fuc_name=""):
    print("="*20+"print info"+"="*20)
    print(f"{get_now_time()}_{fuc_name}, said: {text}")


def save_config(train_config, model_config, data_config, params, save_dir):
    d = {"train_config": train_config, 'model_config': model_config, "data_config": data_config, "params": params}
    save_path = os.path.join(save_dir, "config.json")
    with open(save_path, "w") as fout:
        json.dump(d, fout, ensure_ascii=False, indent=2)


def rmse_score(a, b):
    return np.sqrt(np.mean((a-b)**2))


def mae_score(a,b):
    return np.mean(np.abs(a-b))


class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
                Linear(self.emb_size, self.emb_size),
                ReLU(),
                Dropout(self.dropout),
                Linear(self.emb_size, self.emb_size)
            )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len):
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)


def lt_mask(seq_len):
    return torch.tril(torch.ones(seq_len,seq_len),diagonal=-1).to(dtype=torch.bool).to(device)


def pos_encode(seq_len):
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def cal_meanandstd(x):
    mean = np.mean(x)
    std = np.std(x)
    print(mean, std)


def cosin_similarity(x, y):
    return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))


def generate_qmatrix(data_config, gamma=0.0):
    df_train = pd.read_csv(os.path.join(data_config["dpath"], "train_test_seqs_grp.csv"))[["questions", "concepts"]]
    df_test = pd.read_csv(os.path.join(data_config["dpath"], "train_test_seqs_stu.csv"))[["questions", "concepts"]]
    df = pd.concat([df_train, df_test])

    problem2skill = dict()
    for i, row in df.iterrows():
        qids, cids = [], []
        qids_ori = [[int(i) for i in sublist.split(",")] for sublist in row['questions'].split("|")]
        cids_ori = [[int(i) for i in sublist.split(",")] for sublist in row['concepts'].split("|")]
        for sub_q, sub_c in zip(qids_ori, cids_ori):
            for i, q in enumerate(sub_q):
                if q != -1:
                    qids.append(q)
                    cids.append(sub_c[i])
        assert len(qids) == len(cids), "the qid size must match the cid size!"

        for q, c in zip(qids, cids):
            if q in problem2skill:
                problem2skill[q].append(c)
            else:
                problem2skill[q] = [c]
    print("num ques:{}, num cpts:{}".format(len(set(qids)), len(set(cids))))
    n_problem, n_skill = data_config["num_q"], data_config["num_c"]
    q_matrix = np.zeros((n_problem + 1, n_skill + 1)) + gamma
    for p in problem2skill.keys():
        for c in problem2skill[p]:
            q_matrix[p][c] = 1
    np.savez(os.path.join(data_config["dpath"], "qmatrix.npz"), matrix=q_matrix)
    return q_matrix


class AttentionModule(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0.2):
        super(AttentionModule, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1)
        out = self.linear(x_avg)
        weight = torch.softmax(out.unsqueeze(-1), dim=0)
        output = torch.sum(weight * x, dim=0, keepdim=True)
        a = torch.mean(x, dim=0, keepdim=True)
        return output