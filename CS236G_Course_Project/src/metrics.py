import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


import plotly as plt
import os
import pprint
import argparse

import wandb
import torch
import numpy as np

# Batched CD (CPU), borrowed from https://github.com/ThibaultGROUEIX/AtlasNet
def cd_cpu(sample, ref):
    x, y = sample, ref
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(sample).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]

def compute_cd(x, y, reduce_func=torch.mean):
    d1, d2 = cd_cpu(x, y)
    return reduce_func(d1, dim=1) + reduce_func(d2, dim=1)

def compute_pairwise_cd_emd(x, y, batch_size=32):
    NX, NY, cd, _ = x.size(0), y.size(0), [], []
    y = y.contiguous()
    for i in tqdm(range(NX)):
        cdx, _ , xi = [], [], x[i]
        for j in range(0, NY, batch_size):
            yb = y[j : j + batch_size]
            xb = xi.view(1, -1, 3).expand_as(yb).contiguous()
            cdx.append(compute_cd(xb, yb).view(1, -1))
        cd.append(torch.cat(cdx, dim=1))
    cd = torch.cat(cd, dim=0)
    return cd


def compute_mmd_cov(dxy):
    _, min_idx = dxy.min(dim=1)
    min_val, _ = dxy.min(dim=0)
    mmd = min_val.mean()
    cov = min_idx.unique().numel() / dxy.size(1)
    cov = torch.tensor(cov).to(dxy)
    return mmd, cov


@torch.no_grad()
def compute_metrics(generated_point_cloud, real_point_cloud, batch_size):
    cd_real_gen = compute_pairwise_cd_emd(real_point_cloud, generated_point_cloud, batch_size)
    mmd_cd, cov_cd = compute_mmd_cov(cd_real_gen.t())
    return {
        "COV-CD": cov_cd.cpu(),
        "MMD-CD": mmd_cd.cpu()
    }, {
        "CD_YX": cd_real_gen.cpu()
    }
