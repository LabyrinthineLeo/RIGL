# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math, random
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch_geometric.utils import to_undirected
from torch_geometric.nn import GCNConv
from utils.utils import cosin_similarity, AttentionModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class RIGL(nn.Module):
    def __init__(self, n_question, n_concept,
                 d_model, n_blocks, dropout, d_ff=256, seq_len=200,
                 kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, l2=1e-5, q_matrix=None,
                 emb_type="qid", flip_rate=0.05):
        super().__init__()
        self.model_name = "rigl"
        self.n_question = n_question
        self.n_concept = n_concept
        self.num_c = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.model_type = self.model_name
        self.q_matrix = q_matrix
        self.emb_type = emb_type
        self.n_dim = d_model
        self.flip_rate = flip_rate

        self.ques_embed_layer = nn.Embedding(n_question + 1, d_model)
        self.cpts_embed_layer = nn.Embedding(n_concept + 1, d_model)
        self.qa_embed_layer = nn.Embedding(2 * n_question + 1, d_model)
        self.resp_embed_layer = nn.Embedding(2, d_model)
        self.response_layer = nn.Linear(1, d_model)
        self.AttModule = AttentionModule(d_model)

        self.model = Architecture(n_question=n_question,
                                  n_blocks=n_blocks,
                                  n_heads=num_attn_heads,
                                  dropout=dropout,
                                  d_model=d_model,
                                  d_feature=d_model / num_attn_heads,
                                  d_ff=d_ff,
                                  kq_same=self.kq_same,
                                  model_type=self.model_type,
                                  seq_len=seq_len)

        self.graph_layer1 = GCNConv(2*d_model, 128)
        self.graph_layer2 = GCNConv(128, 2*d_model)

        self.out_layer = nn.Sequential(
            nn.Linear(2 * d_model, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, n_question)
        )

    def base_emb(self, ques, resp, mode):
        ques_embed = self.ques_embed_layer(ques)
        cpts = torch.argmax(torch.Tensor(self.q_matrix[ques.detach().cpu().numpy()]).to(device), dim=-1, keepdim=True).to(device)
        cpts_embed = self.cpts_embed_layer(cpts).squeeze(-2)
        q_embed = ques_embed + cpts_embed
        if mode == "stu":
            qa_embed = self.resp_embed_layer(resp) + ques_embed
        else:
            qa_embed = self.response_layer(resp.unsqueeze(-1)) + ques_embed

        return q_embed, qa_embed

    def sim_stu(self, x1, x2, proj=None):
        n_stu, n_span, _ = x1.size()
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(0)
        if proj is not None:
            x1 = proj(x1)
            x2 = proj(x2)
        return F.cosine_similarity(x1, x2, dim=-1) / 0.05

    def flip_response(self, resp, mask):
        n_stu, n_span, n_seq = mask.size()
        resp_ = resp.clone()
        for stu in range(n_stu):
            for span in range(n_span):
                # manipulate score
                seq_len = int(mask[stu][span].sum())
                if seq_len == 0:
                    continue
                idx = random.sample(
                    range(seq_len), max(1, int(seq_len * self.flip_rate))
                )
                for i in idx:
                    resp_[stu, span, i] = 1 - resp_[stu, span, i]
        return resp_

    def forward(self, qids_grp, resp_grp, mask_grp, qids_stu_ori, resp_stu_ori, mask_stu_ori, CL=False, top_num=3):
        n_grp, n_timeframe = qids_grp.shape[0], qids_grp.shape[1]
        num_stu = np.count_nonzero(mask_stu_ori.cpu().numpy().any(axis=(1, 2)))
        qids_stu = qids_stu_ori[:num_stu]
        resp_stu = resp_stu_ori[:num_stu]
        mask_stu = mask_stu_ori[:num_stu]
        if CL:
            resp_stu = self.flip_response(resp_stu, mask_stu)

        q_embed_stu, qa_embed_stu = self.base_emb(qids_stu, resp_stu, "stu")
        q_embed_stu_ori = (q_embed_stu * mask_stu.unsqueeze(-1)).mean(dim=2)
        qa_embed_stu_ori = (qa_embed_stu * mask_stu.unsqueeze(-1)).mean(dim=2)

        q_embed_grp, qa_embed_grp = self.base_emb(qids_grp, resp_grp, "grp")
        q_embed_grp_ori = (q_embed_grp * mask_grp.unsqueeze(-1)).mean(dim=2)
        qa_embed_grp_ori = (qa_embed_grp * mask_grp.unsqueeze(-1)).mean(dim=2)

        q_embed_grp = q_embed_grp_ori + self.AttModule(q_embed_stu_ori)
        qa_embed_grp = qa_embed_grp_ori + self.AttModule(qa_embed_stu_ori)
        q_embed_stu = q_embed_stu_ori + q_embed_grp_ori
        qa_embed_stu = qa_embed_stu_ori + qa_embed_grp_ori

        grp_node_id = 0
        stu_node_ids = [i+1 for i in range(num_stu)]
        embed_grp_graph = torch.zeros(n_grp, n_timeframe, 2*self.n_dim).to(device)
        embed_stu_graph = torch.zeros(num_stu, n_timeframe, 2*self.n_dim).to(device)
        for t in range(n_timeframe):
            t_q_embed_grp = q_embed_grp[:, t, :]
            t_qa_embed_grp = qa_embed_grp[:, t, :]
            t_embed_grp = torch.cat((t_q_embed_grp, t_qa_embed_grp), dim=1)
            t_q_embed_stu = q_embed_stu[:, t, :]
            t_qa_embed_stu = qa_embed_stu[:, t, :]
            t_embed_stu = torch.cat((t_q_embed_stu, t_qa_embed_stu), dim=1)

            source_nodes = [grp_node_id] * num_stu
            target_nodes = [i for i in stu_node_ids]

            for x_id in range(num_stu):
                sims = []
                for y_id in range(num_stu):
                    if x_id != y_id:
                        sim = cosin_similarity(t_embed_stu[x_id], t_embed_stu[y_id])
                        sims.append((y_id, sim.item()))

                sims.sort(key=lambda x: x[1], reverse=True)
                top_sims = sims[:top_num]
                source_nodes.extend([x_id+1] * len(top_sims))
                target_nodes.extend([i[0]+1 for i in top_sims])

            edge_index = to_undirected(torch.LongTensor([source_nodes, target_nodes]).to(device))
            x = torch.cat((t_embed_grp, t_embed_stu), dim=0)

            out = self.graph_layer1(x, edge_index)
            out = F.relu(out)
            out = F.dropout(out, training=self.training)
            out = self.graph_layer2(out, edge_index)
            embed_grp_graph[:, t, :] = out[0]
            embed_stu_graph[:, t, :] = out[1:]

        q_embed_grp_graph = embed_grp_graph[:, :, :self.n_dim]
        qa_embed_grp_graph = embed_grp_graph[:, :, self.n_dim:]
        q_embed_stu_graph = embed_stu_graph[:, :, :self.n_dim]
        qa_embed_stu_graph = embed_stu_graph[:, :, self.n_dim:]

        output_stu = self.model(q_embed_stu_graph, qa_embed_stu_graph)
        output_grp = self.model(q_embed_grp_graph, qa_embed_grp_graph)

        concat_stu = torch.cat([output_stu, q_embed_stu_graph], dim=-1)
        concat_grp = torch.cat([output_grp, q_embed_grp_graph], dim=-1)
        stu_output = self.out_layer(concat_stu).squeeze(-1)
        grp_output = self.out_layer(concat_grp).squeeze(-1)
        out_stu = nn.Sigmoid()(stu_output)
        out_grp = nn.Sigmoid()(grp_output)
        return out_grp, out_stu

    def get_cl_loss(self, qids_grp, resp_grp, mask_grp, qids_stu_ori, resp_stu_ori, mask_stu_ori, top_num=3):
        out_grp, out_stu = self(qids_grp, resp_grp, mask_grp, qids_stu_ori, resp_stu_ori, mask_stu_ori)
        out_grp_cl, out_stu_cl = self(qids_grp, resp_grp, mask_grp, qids_stu_ori, resp_stu_ori, mask_stu_ori, True)

        n_stu, n_span = out_stu.shape[0], out_stu.shape[1]
        input = self.sim_stu(out_stu, out_stu_cl)
        target = (torch.arange(n_stu)[:, None].to(device).expand(-1, n_span))
        cl_loss = F.cross_entropy(input, target)
        return out_grp, out_stu, cl_loss


class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
            for _ in range(n_blocks)
        ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder

        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y,
                      apply_pos=True)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask,
                zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
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

    def forward(self, q, k, v, mask, zero_pad):

        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad)

        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
             math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]