# -*- coding: utf-8 -*-
import os, sys
sys.path.append("..")

import logging
import json

import torch
from torch.optim import SGD, Adam
import copy

from utils import set_seed, debug_print, save_config
from .dataset import dataset4train
from .init_model import init_model
from .train_model import train_model
import datetime
import numpy as np

device = "cpu" if not torch.cuda.is_available() else "cuda"


def main(params):
    if "use_wandb" not in params:
        params['use_wandb'] = 0
    set_seed(params["seed"])
    model_name, dataset_name, fold, emb_type, save_dir = params["model_name"], params["dataset_name"], \
                                                         params["fold"], params["emb_type"], params["save_dir"]

    debug_print(text="load config files.", fuc_name="main")
    train_config = {
        "batch_size": 1 if params["mode"] == "both" else 256,
        "num_epochs": 120,
        "optimizer": "adam",
        "seq_len": params['seq_len']
    }
    model_config = copy.deepcopy(params)
    for key in ["model_name", "dataset_name", "emb_type", "save_dir", "fold", "seed"]:
        del model_config[key]
    if 'batch_size' in params:
        train_config["batch_size"] = params['batch_size']
    if 'num_epochs' in params:
        train_config["num_epochs"] = params['num_epochs']

    batch_size, num_epochs, optimizer = train_config["batch_size"], train_config["num_epochs"], train_config["optimizer"]

    with open("../configs/data_config.json") as fin:
        data_config = json.load(fin)

    seq_len = train_config["seq_len"]

    print("=" * 20 + "start init data" + "=" * 20)
    print(dataset_name, model_name)
    for i, j in data_config.items():
        print(f"{i}: {j}")
    print(fold, batch_size)

    debug_print(text="init_dataset", fuc_name="main")
    mode = params["mode"]
    train_loader_list, test_loader_list, *_ = dataset4train(dataset_name, model_name, data_config, seq_len, batch_size, mode)

    params_str = "_".join([str(v) for k, v in params.items() if not k in ['other_config']])
    if params['add_uuid'] == 1 and params["use_wandb"] == 1:
        import uuid
        params_str = params_str + f"_{str(uuid.uuid4())}"
    time_suffix = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    result_path = os.path.join(save_dir, params_str+f"_{time_suffix}")
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    start_time = datetime.datetime.now().timestamp()
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    logging.basicConfig(
        filename=os.path.join(result_path, 'log.txt'),
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        format='[%(asctime)s %(levelname)s] %(message)s',
    )
    print("=" * 20 + "print params" + "=" * 20)
    logging.info("=" * 20 + "print params" + "=" * 20)
    for i, j in params.items():
        print(f"{i}: {j}")
        logging.info(f"{i}: {j}")

    print("=" * 20 + "training model" + "=" * 20)
    logging.info("=" * 20 + "training model" + "=" * 20)
    print(f"Start training model: {model_name}, \nembtype: {emb_type}, \nsave_dir: {result_path}, \ndataset_name: {dataset_name}")
    logging.info(f"Start training model: {model_name}, \nembtype: {emb_type}, \nsave_dir: {result_path}, \ndataset_name: {dataset_name}")
    print("=" * 20 + "model config" + "=" * 20)
    logging.info("=" * 20 + "model config" + "=" * 20)
    print(f"model_config: {model_config}")
    logging.info(f"model_config: {model_config}")
    print("=" * 20 + "train config" + "=" * 20)
    logging.info("=" * 20 + "train config" + "=" * 20)
    print(f"train_config: {train_config}")
    logging.info(f"train_config: {train_config}")

    save_config(train_config, model_config, data_config[dataset_name], params, result_path)
    learning_rate = params["learning_rate"]
    for remove_item in ['use_wandb', 'learning_rate', 'add_uuid', 'l2', 'seq_len', 'mode']:
        if remove_item in model_config:
            del model_config[remove_item]

    if model_name in ["rigl"]:
        model_config["seq_len"] = seq_len

    debug_print(text="init_model", fuc_name="main")

    print("=" * 20 + "init model" + "=" * 20)
    logging.info("=" * 20 + "init model" + "=" * 20)
    print(f"model_name:{model_name}")
    logging.info(f"model_name:{model_name}")

    model = init_model(model_name, model_config, data_config[dataset_name], emb_type, mode)
    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    save_model = True

    debug_print(text="train model", fuc_name="main")

    test_rmse_list, test_mae_list = [], []
    test_auc_list, test_acc_list = [], []
    for i, (train_loader, test_loader) in enumerate(zip(train_loader_list, test_loader_list)):
        print("=" * 20 + "training time:{}".format(i) + "=" * 20)
        if mode == "both":
            testauc, testacc, testrmse, testmae, best_epoch = train_model(model, train_loader, test_loader, num_epochs, opt, result_path, mode, None, None, save_model)

        model = init_model(model_name, model_config, data_config[dataset_name], emb_type, mode)
        if optimizer == "sgd":
            opt = SGD(model.parameters(), learning_rate, momentum=0.9)
        elif optimizer == "adam":
            opt = Adam(model.parameters(), learning_rate)

        if mode == "both":
            test_rmse_list.append(testrmse)
            test_mae_list.append(testmae)

        test_auc_list.append(testauc)
        test_acc_list.append(testacc)
    print("=" * 20 + "auc list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_auc_list])
    logging.info(["{:.5}".format(i) for i in test_auc_list])
    print("=" * 20 + "acc list" + "=" * 20)
    print(["{:.5}".format(i) for i in test_acc_list])
    logging.info(["{:.5}".format(i) for i in test_acc_list])
    if mode == "both":
        print("=" * 20 + "rmse list" + "=" * 20)
        print(["{:.5}".format(i) for i in test_rmse_list])
        logging.info(["{:.5}".format(i) for i in test_rmse_list])
        print("=" * 20 + "mae list" + "=" * 20)
        print(["{:.5}".format(i) for i in test_mae_list])
        logging.info(["{:.5}".format(i) for i in test_mae_list])

    print("=" * 20 + "the mean and std of auc/acc" + "=" * 20)
    print("auc mean:{:.5}, auc std:{:.5}".format(np.mean(test_auc_list), np.std(test_auc_list)))
    logging.info("auc mean:{:.5}, auc std:{:.5}".format(np.mean(test_auc_list), np.std(test_auc_list)))
    print("acc mean:{:.5}, acc std:{:.5}".format(np.mean(test_acc_list), np.std(test_acc_list)))
    logging.info("acc mean:{:.5}, acc std:{:.5}".format(np.mean(test_acc_list), np.std(test_acc_list)))

    if mode == "both":
        print("=" * 20 + "the mean and std of rmse/mae" + "=" * 20)
        print("rmse mean:{:.5}, rmse std:{:.5}".format(np.mean(test_rmse_list), np.std(test_rmse_list)))
        logging.info("rmse mean:{:.5}, rmse std:{:.5}".format(np.mean(test_rmse_list), np.std(test_rmse_list)))
        print("mae mean:{:.5}, mae std:{:.5}".format(np.mean(test_mae_list), np.std(test_mae_list)))
        logging.info("mae mean:{:.5}, mae std:{:.5}".format(np.mean(test_mae_list), np.std(test_mae_list)))

    print(f"start:{now}")
    print(f"end:{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    end_time = datetime.datetime.now().timestamp()
    print(f"cost time:{(end_time-start_time)//60} min")