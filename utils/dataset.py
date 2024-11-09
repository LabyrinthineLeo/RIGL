# -*- coding: utf-8 -*-

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
else:
    from torch import FloatTensor, LongTensor


class HKTDataset(Dataset):
    def __init__(self, csv_path_grp, csv_path_stu, input_type, folds):
        super(HKTDataset, self).__init__()
        seq_path_grp = csv_path_grp
        seq_path_stu = csv_path_stu
        self.seq_grp_keys = ["qseqs_grp", "cseqs_grp", "rseqs_grp", "tseqs_grp"]
        self.seq_stu_keys = ["qseqs_stu", "cseqs_stu", "rseqs_stu", "tseqs_stu"]
        self.input_type = input_type
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(_) for _ in folds])
        processed_data_both = csv_path_grp + folds_str + "_grp_stu.pkl"

        if os.path.exists(processed_data_both):
            print("=" * 20 + "reading data from pkl" + "=" * 20)
            print(f"Read data from processed file: {processed_data_both}")
            if torch.cuda.is_available():
                self.dict_both = pd.read_pickle(processed_data_both)
            else:
                self.dict_both = pd.read_pickle(processed_data_both)

            for key in self.dict_both:
                self.dict_both[key] = self.dict_both[key]
        else:
            print("=" * 20 + "precessing both data to pkl" + "=" * 20)
            self.dict_both = self.__load_both_data__(seq_path_grp, seq_path_stu, folds)
            save_data = self.dict_both
            pd.to_pickle(save_data, processed_data_both)

        print("=" * 20 + "start load both dataset" + "=" * 20)

    def __len__(self):
        return len(self.dict_both["rseqs_both"])

    def __getitem__(self, index):
        dict_cur = dict()
        for key in self.dict_both:
            if key in ["pool_smasks_both", "pred_smasks_both"]:
                continue
            if len(self.dict_both[key]) == 0:
                dict_cur[key] = self.dict_both[key]
                dict_cur["shft_" + key] = self.dict_both[key]
                continue

            seqs_both = self.dict_both[key][index][:, :-1, :] * self.dict_both["pool_smasks_both"][index]
            dict_cur[key] = seqs_both

            shift_seqs_both = self.dict_both[key][index][:, 1:, :] * self.dict_both["pred_smasks_both"][index]
            dict_cur["shft_" + key] = shift_seqs_both

        dict_cur["pool_smasks_both"] = self.dict_both["pool_smasks_both"][index]
        dict_cur["pred_smasks_both"] = self.dict_both["pred_smasks_both"][index]

        return dict_cur

    def __load_both_data__(self, seq_path_grp, seq_path_stu, folds, pad_val=-1):
        dict_both = {"qseqs_both": [], "cseqs_both": [], "rseqs_both": [], "smasks_both": []}

        df_grp = pd.read_csv(seq_path_grp)
        df_grp = df_grp[df_grp["fold"].isin(folds)]
        df_stu = pd.read_csv(seq_path_stu)
        stu_num_list = []
        for idx, each_group in df_stu.groupby(["gid"], sort=False):
            stu_num_list.append(len(each_group))
            if idx == (0,):  # new version of the index value
                res_list = [[i for i in sublist.split(",")] for sublist in each_group["responses"].tolist()[0].split("|")]
                num_span = len(res_list)
                num_seq = len(res_list[0])
        max_stu_num = max(stu_num_list)

        for idx, (x, row) in enumerate(df_grp.iterrows()):
            both_list_cpt = []
            both_list_que = []
            both_list_res = []
            both_list_sm = []

            gid = row["gid"]
            each_group = df_stu[df_stu["gid"] == gid]
            stu_num = len(each_group)

            both_list_cpt.append([[int(i) for i in sublist.split(",")] for sublist in row['concepts'].split("|")])
            both_list_cpt.extend([[[int(i) for i in sublist.split(",")] for sublist in each_cpt.split("|")] for each_cpt in each_group['concepts'].tolist()])
            for k in range(max_stu_num-stu_num):
                both_list_cpt.append([[-1] * num_seq for _ in range(num_span)])
            dict_both['cseqs_both'].append(both_list_cpt)

            both_list_que.append([[int(i) for i in sublist.split(",")] for sublist in row['questions'].split("|")])
            both_list_que.extend([[[int(i) for i in sublist.split(",")] for sublist in each_que.split("|")] for each_que in each_group['questions'].tolist()])
            for k in range(max_stu_num - stu_num):
                both_list_que.append([[-1] * num_seq for _ in range(num_span)])
            dict_both['qseqs_both'].append(both_list_que)

            both_list_res.append([[float(i) for i in sublist.split(",")] for sublist in row['responses'].split("|")])
            both_list_res.extend([[[int(i) for i in sublist.split(",")] for sublist in each_res.split("|")] for each_res in each_group['responses'].tolist()])
            for k in range(max_stu_num - stu_num):
                both_list_res.append([[-1] * num_seq for _ in range(num_span)])
            dict_both['rseqs_both'].append(both_list_res)

            both_list_sm.append([[int(i) for i in sublist.split(",")] for sublist in row['selectmasks'].split("|")])
            both_list_sm.extend([[[int(i) for i in sublist.split(",")] for sublist in each_sm.split("|")] for each_sm in each_group['selectmasks'].tolist()])
            for k in range(max_stu_num - stu_num):
                both_list_sm.append([[-1] * num_seq for _ in range(num_span)])
            dict_both['smasks_both'].append(both_list_sm)

        for key in dict_both:
            if key not in ["rseqs_both", "rseqs_grp", "rseqs_stu"]:
                dict_both[key] = LongTensor(dict_both[key])
            else:
                dict_both[key] = FloatTensor(dict_both[key])

        dict_both["pool_smasks_both"] = (dict_both["smasks_both"][:, :, :-1, :] != pad_val)
        dict_both["pred_smasks_both"] = (dict_both["smasks_both"][:, :, 1:, :] != pad_val)

        return dict_both


def dataset4train(dataset_name, model_name, data_config, maxlen, batch_size, mode="both"):
    data_config = data_config[dataset_name]
    fold_list = set(data_config["folds"])
    trainset_list = []
    testset_list = []
    for i in fold_list:
        train = HKTDataset(os.path.join(data_config["dpath"], data_config[f"train_test_seq_grp"]),
                          os.path.join(data_config["dpath"], data_config[f"train_test_seq_stu"]),
                          data_config["input_type"], fold_list - {i})
        test = HKTDataset(os.path.join(data_config["dpath"], data_config[f"train_test_seq_grp"]),
                         os.path.join(data_config["dpath"], data_config[f"train_test_seq_stu"]),
                         data_config["input_type"], {i})
        trainset_list.append(train)
        testset_list.append(test)

    train_loader_list = [DataLoader(trn, batch_size=batch_size) for trn in trainset_list]
    test_loader_list = [DataLoader(tst, batch_size=batch_size) for tst in testset_list]

    return train_loader_list, test_loader_list