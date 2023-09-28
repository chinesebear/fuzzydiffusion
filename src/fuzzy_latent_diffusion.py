import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from loguru import logger
from tqdm import tqdm

from lssim import latent_structural_similarity

class FuzzyLatentDiffusion(pl.LightningModule):
    def __init__(self, rule_num, delegates, ldmodels):
        self.rule_num = rule_num
        self.delegates = delegates
        self.ldms = nn.ModuleList(ldmodels)
    def membership(self, x, delegate):
        latent_ssim_score = latent_structural_similarity(x, delegate, data_range=255, multichannel=True)
        return latent_ssim_score
    def antecedent(self, x):
        # membership array
        u_arr = torch.tensor()
        for i in range(self.rule_num):
            u = self.membership(x, self.delegates[i])
            u_arr.append(u)
        # normalization
        fires = u_arr
        max = fires
        return fires
    def consequent(self, fires, x):
        x_out = x
        for i in range(self.rule_num):
            x_out = x_out + fires[i].self.ldms[i](x)
        return x_out
    def forward(self, data):
        fires = self.antecedent(data)
        output = self.consequent(fires, data)
        return output
    
