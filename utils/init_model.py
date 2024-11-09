# -*- coding: utf-8 -*-

import torch, os
import numpy as np
from utils import generate_qmatrix
from models.RIGL import RIGL

device = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model(model_name, model_config, data_config, emb_type, mode="both"):
    assert model_name == "rigl", "The proposed RIGL model."
    qmatrix_path = os.path.join(data_config["dpath"], "qmatrix.npz")
    if os.path.exists(qmatrix_path):
        q_matrix = np.load(qmatrix_path, allow_pickle=True)['matrix']
    else:
        q_matrix = generate_qmatrix(data_config)

    model = RIGL(data_config["num_q"], data_config["num_c"], **model_config, q_matrix=q_matrix, emb_type=emb_type).to(device)

    return model