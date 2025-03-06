import torch
import os
import numpy as np
import random
from loguru import logger


class Options:
    def __init__(self, name) -> None:
        self.name= name
    def name(self):
        return self.name

# project gloal parameter
options = Options("Model")
options.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
options.evaluate_path="/home/yang/sda/github/evaluate/"
options.base_path="/home/yang/sda/github/fuzzydiffusion/"
options.model_parameter_path = options.base_path+"output/pth/"
options.dataset_path = options.base_path+"output/datasets/"
options.img_path = options.base_path+"output/img/"
options.seed_id = 10
options.epoch= 80 #200
options.learning_rate = 1e-4
options.T = 1000
options.batch_size = 8 #80
options.img_width = 128 #32
options.img_height = 128 #32
options.img_size = (options.img_height,options.img_width)
options.multiplier = 2.0
options.grad_clip = 1.0
options.SOS = 0
options.EOS = 1
options.PAD = 2
options.UNK = 3

unet = Options("UNet")
unet.channel =  128 #128
unet.channel_mult = [1, 2, 3, 4]
unet.attn = [2]
unet.num_res_blocks = 2
unet.dropout = 0.15
options.unet = unet

diff = Options("diffusion")
diff.beta_1 = 1e-4
diff.beta_T = 0.02
options.diff = diff

def setting_info():
    output = ""
    output += "epoch: "+ str(options.epoch)+", "
    output += "T: "+ str(options.T)+", "
    output += "batch_size: "+ str(options.batch_size)+", "
    output += "img_size: "+ str(options.img_width)+"*"+ str(options.img_height)+", "
    output += "drop_out: "+ str(options.unet.dropout)+", "
    output += "learning_rate: "+ str(options.learning_rate)
    return output
