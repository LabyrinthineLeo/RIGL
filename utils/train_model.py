# -*- coding: utf-8 -*-

import os
import torch
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy, mse_loss
import numpy as np
from sklearn import metrics
from .utils import debug_print, rmse_score, mae_score
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cal_loss_both(model, ys, r, rshft, sm, n_stu, preloss=[]):
    model_name = model.model_name
    assert n_stu >= 1, "the num of students must larger than 1!"
    gamma = 0.001
    assert model_name == "rigl", "The proposed RIGL model."
    stu_num = ys[1].shape[0]
    y_grp = torch.masked_select(ys[0], sm[0])
    t_grp = torch.masked_select(rshft[0], sm[0])
    y_stu = torch.masked_select(ys[1], sm[1])
    t_stu = torch.masked_select(rshft[1], sm[1])
    loss_grp = mse_loss(y_grp.double(), t_grp.double())
    loss_stu = binary_cross_entropy(y_stu.double(), t_stu.double())
    loss = loss_grp + loss_stu / stu_num + gamma * preloss[0]

    return loss


def model_forward(model, data, mode):
    model_name = model.model_name
    dcur = data
    ques_grp, cpts_grp, reps_grp = dcur["qseqs_both"][:, 0:1, :, :], dcur["cseqs_both"][:, 0:1, :, :], dcur["rseqs_both"][:, 0:1, :, :]
    ques_stu, cpts_stu, reps_stu = dcur["qseqs_both"][:, 1:, :, :], dcur["cseqs_both"][:, 1:, :, :], dcur["rseqs_both"][:, 1:, :, :]
    ques_shft_grp, cpts_shft_grp, reps_shft_grp = dcur["shft_qseqs_both"][:, 0:1, :, :], dcur["shft_cseqs_both"][:, 0:1, :, :], dcur["shft_rseqs_both"][:, 0:1, :, :]
    ques_shft_stu, cpts_shft_stu, reps_shft_stu = dcur["shft_qseqs_both"][:, 1:, :, :], dcur["shft_cseqs_both"][:, 1:, :, :], dcur["shft_rseqs_both"][:, 1:, :, :]

    assert ques_grp.shape[0] == 1, "Currently only supports loading by each group"

    ques_grp, cpts_grp, reps_grp = ques_grp.squeeze(0), cpts_grp.squeeze(0), reps_grp.squeeze(0)
    ques_stu, cpts_stu, reps_stu = ques_stu.squeeze(0), cpts_stu.squeeze(0), reps_stu.squeeze(0)
    ques_shft_grp, cpts_shft_grp, reps_shft_grp = ques_shft_grp.squeeze(0), cpts_shft_grp.squeeze(0), reps_shft_grp.squeeze(0)
    ques_shft_stu, cpts_shft_stu, reps_shft_stu = ques_shft_stu.squeeze(0), cpts_shft_stu.squeeze(0), reps_shft_stu.squeeze(0)

    n_seq = ques_grp.shape[2]
    pool_sm_grp = dcur["pool_smasks_both"][:, 0:1, :, :].squeeze(0)
    pool_sm_stu = dcur["pool_smasks_both"][:, 1:, :, :].squeeze(0)
    pred_sm_grp = dcur["pred_smasks_both"][:, 0:1, :, :].squeeze(0)
    pred_sm_stu = dcur["pred_smasks_both"][:, 1:, :, :].squeeze(0)

    ques_grp = ques_grp.to(device)
    ques_stu = ques_stu.to(device)
    cpts_grp = cpts_grp.to(device)
    cpts_stu = cpts_stu.to(device)
    reps_grp = reps_grp.to(device)
    reps_stu = reps_stu.to(device)
    ques_shft_grp = ques_shft_grp.to(device)
    ques_shft_stu = ques_shft_stu.to(device)
    cpts_shft_grp = cpts_shft_grp.to(device)
    cpts_shft_stu = cpts_shft_stu.to(device)
    reps_shft_grp = reps_shft_grp.to(device)
    reps_shft_stu = reps_shft_stu.to(device)
    pool_sm_grp = pool_sm_grp.to(device)
    pool_sm_stu = pool_sm_stu.to(device)
    pred_sm_grp = pred_sm_grp.to(device)
    pred_sm_stu = pred_sm_stu.to(device)

    if model_name == "rigl":
        num_stu = np.count_nonzero(pool_sm_stu.cpu().numpy().any(axis=(1, 2)))
        cpts_stu = cpts_stu[:num_stu]
        cpts_shft_stu = cpts_shft_stu[:num_stu]
        ques_stu = ques_stu[:num_stu]
        ques_shft_stu = ques_shft_stu[:num_stu]
        reps_stu = reps_stu[:num_stu]
        reps_shft_stu = reps_shft_stu[:num_stu]
        pool_sm_stu = pool_sm_stu[:num_stu]
        pred_sm_stu = pred_sm_stu[:num_stu]

    y_grp_stu, ys, preloss = [], [], []

    cq_grp = torch.cat((ques_grp[:, 0:1, :], ques_shft_grp), dim=1)
    cc_grp = torch.cat((cpts_grp[:, 0:1, :], cpts_shft_grp), dim=1)
    cr_grp = torch.cat((reps_grp[:, 0:1, :], reps_shft_grp), dim=1)
    cat_sm_grp = torch.cat((pool_sm_grp[:, 0:1, :], pred_sm_grp), dim=1)

    cq_stu = torch.cat((ques_stu[:, 0:1, :], ques_shft_stu), dim=1)
    cc_stu = torch.cat((cpts_stu[:, 0:1, :], cpts_shft_stu), dim=1)
    cr_stu = torch.cat((reps_stu[:, 0:1, :], reps_shft_stu), dim=1)
    cat_sm_stu = torch.cat((pool_sm_stu[:, 0:1, :], pred_sm_stu), dim=1)
    num_stu = np.count_nonzero(cat_sm_stu.cpu().numpy().any(axis=(1, 2)))

    assert model_name == "rigl", "The proposed RIGL model."
    if model_name == "rigl":
        y_grp, y_stu, cl_loss = model.get_cl_loss(cq_grp.long(), cr_grp, cat_sm_grp, cq_stu.long(), cr_stu.long(), cat_sm_stu)
        y_grp = y_grp[:, :-1, :]
        y_grp = (y_grp.unsqueeze(2).repeat(1, 1, n_seq, 1) * one_hot(ques_shft_grp.long(), model.num_c)).sum(-1)
        y_stu = y_stu[:, :-1, :]
        y_stu = (y_stu.unsqueeze(2).repeat(1, 1, n_seq, 1) * one_hot(ques_shft_stu.long(), model.num_c)).sum(-1)

        y_grp_stu.append(y_grp)
        y_grp_stu.append(y_stu)
        preloss.append(cl_loss)

    loss = cal_loss_both(model, y_grp_stu, [reps_grp, reps_stu], [reps_shft_grp, reps_shft_stu],
                                 [pred_sm_grp, pred_sm_stu], num_stu, preloss)

    return loss


def evaluate_both(model, test_loader, model_name, save_path=""):
    if save_path != "":
        fout = open(save_path, "w", encoding="utf8")
    with torch.no_grad():
        y_grp_trues = []
        y_stu_trues = []
        y_grp_scores = []
        y_stu_scores = []
        dres = dict()
        test_mini_index = 0
        for data in test_loader:
            dcur = data

            ques_grp, cpts_grp, reps_grp = dcur["qseqs_both"][:, 0:1, :, :].to(device), dcur["cseqs_both"][:, 0:1, :, :].to(device), dcur["rseqs_both"][:, 0:1, :, :].to(device)
            ques_stu, cpts_stu, reps_stu = dcur["qseqs_both"][:, 1:, :, :].to(device), dcur["cseqs_both"][:, 1:, :, :].to(device), dcur["rseqs_both"][:, 1:, :, :].to(device)
            ques_shft_grp, cpts_shft_grp, reps_shft_grp = dcur["shft_qseqs_both"][:, 0:1, :, :].to(device), dcur["shft_cseqs_both"][:, 0:1, :, :].to(device), dcur["shft_rseqs_both"][:, 0:1, :, :].to(device)
            ques_shft_stu, cpts_shft_stu, reps_shft_stu = dcur["shft_qseqs_both"][:, 1:, :, :].to(device), dcur["shft_cseqs_both"][:, 1:, :, :].to(device), dcur["shft_rseqs_both"][:, 1:, :, :].to(device)

            assert ques_grp.shape[0] == 1, "Currently only supports loading by each group"

            ques_grp, cpts_grp, reps_grp = ques_grp.squeeze(0).to(device), cpts_grp.squeeze(0).to(device), reps_grp.squeeze(0).to(device)
            ques_stu, cpts_stu, reps_stu = ques_stu.squeeze(0).to(device), cpts_stu.squeeze(0).to(device), reps_stu.squeeze(0).to(device)
            ques_shft_grp, cpts_shft_grp, reps_shft_grp = ques_shft_grp.squeeze(0).to(device), cpts_shft_grp.squeeze(0).to(device), reps_shft_grp.squeeze(0).to(device)
            ques_shft_stu, cpts_shft_stu, reps_shft_stu = ques_shft_stu.squeeze(0).to(device), cpts_shft_stu.squeeze(0).to(device), reps_shft_stu.squeeze(0).to(device)

            n_seq = ques_grp.shape[2]
            pool_sm_grp = dcur["pool_smasks_both"][:, 0:1, :, :].squeeze(0).to(device)
            pool_sm_stu = dcur["pool_smasks_both"][:, 1:, :, :].squeeze(0).to(device)
            pred_sm_grp = dcur["pred_smasks_both"][:, 0:1, :, :].squeeze(0).to(device)
            pred_sm_stu = dcur["pred_smasks_both"][:, 1:, :, :].squeeze(0).to(device)

            num_stu = np.count_nonzero(pool_sm_stu.cpu().numpy().any(axis=(1, 2)))
            cpts_stu = cpts_stu[:num_stu]
            cpts_shft_stu = cpts_shft_stu[:num_stu]
            ques_stu = ques_stu[:num_stu]
            ques_shft_stu = ques_shft_stu[:num_stu]
            reps_stu = reps_stu[:num_stu]
            reps_shft_stu = reps_shft_stu[:num_stu]
            pool_sm_stu = pool_sm_stu[:num_stu]
            pred_sm_stu = pred_sm_stu[:num_stu]

            cq_grp = torch.cat((ques_grp[:, 0:1, :], ques_shft_grp), dim=1)
            cq_stu = torch.cat((ques_stu[:, 0:1, :], ques_shft_stu), dim=1)
            cc_grp = torch.cat((cpts_grp[:, 0:1, :], cpts_shft_grp), dim=1)
            cc_stu = torch.cat((cpts_stu[:, 0:1, :], cpts_shft_stu), dim=1)
            cr_grp = torch.cat((reps_grp[:, 0:1, :], reps_shft_grp), dim=1)
            cr_stu = torch.cat((reps_stu[:, 0:1, :], reps_shft_stu), dim=1)

            cat_sm_grp = torch.cat((pool_sm_grp[:, 0:1, :], pred_sm_grp), dim=1)
            cat_sm_stu = torch.cat((pool_sm_stu[:, 0:1, :], pred_sm_stu), dim=1)

            model.eval()

            y_grp, y_stu = model(cq_grp.long(), cr_grp, cat_sm_grp, cq_stu.long(), cr_stu.long(), cat_sm_stu)
            y_grp = y_grp[:, :-1, :]
            y_grp = (y_grp.unsqueeze(2).repeat(1, 1, n_seq, 1) * one_hot(ques_shft_grp.long(), model.num_c)).sum(-1)
            y_stu = y_stu[:, :-1, :]
            y_stu = (y_stu.unsqueeze(2).repeat(1, 1, n_seq, 1) * one_hot(ques_shft_stu.long(), model.num_c)).sum(-1)

            y_grp = torch.masked_select(y_grp, pred_sm_grp).detach().cpu()
            t_grp = torch.masked_select(reps_shft_grp, pred_sm_grp).detach().cpu()
            y_stu = torch.masked_select(y_stu, pred_sm_stu).detach().cpu()
            t_stu = torch.masked_select(reps_shft_stu, pred_sm_stu).detach().cpu()

            y_grp_trues.append(t_grp.numpy())
            y_stu_trues.append(t_stu.numpy())
            y_grp_scores.append(y_grp.numpy())
            y_stu_scores.append(y_stu.numpy())
            test_mini_index += 1

        ts_grp = np.concatenate(y_grp_trues, axis=0)
        ts_stu = np.concatenate(y_stu_trues, axis=0)
        ps_grp = np.concatenate(y_grp_scores, axis=0)
        ps_stu = np.concatenate(y_stu_scores, axis=0)
        auc = metrics.roc_auc_score(y_true=ts_stu, y_score=ps_stu)
        rmse = rmse_score(ts_grp, ps_grp)
        mae = mae_score(ts_grp, ps_grp)

        prelabels = [1 if p >= 0.5 else 0 for p in ps_stu]
        acc = metrics.accuracy_score(ts_stu, prelabels)
    return acc, auc, rmse, mae


def train_model(model, train_loader, valid_loader, num_epochs, opt, result_path, mode="only_stu", test_loader=None, test_window_loader=None, save_model=False):
    min_rmse, max_auc, best_epoch = np.inf, 0, -1
    train_step = 0

    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step += 1
            model.train()

            loss = model_forward(model, data, mode)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())
        loss_mean = np.mean(loss_mean)

        acc, auc, rmse, mae = evaluate_both(model, valid_loader, model.model_name)

        if auc > max_auc:
            if save_model:
                torch.save(model.state_dict(), os.path.join(result_path, model.emb_type + "_model.ckpt"))
            max_auc = auc
            best_epoch = i

        validauc, validacc, validrmse, validmae = auc, acc, rmse, mae

        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, validrmse: {validrmse:.4}, validmae: {validmae:.4}, "
              f"best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean:.5}, "
              f"emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {result_path.split('_')[-1]}")
        logging.info(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, validrmse: {validrmse:.4}, validmae: {validmae:.4}, "
                     f"best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean:.5}, "
                     f"emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {result_path.split('_')[-1]}")
        if i - best_epoch >= 30:
            break

    return validauc, validacc, validrmse, validmae, best_epoch