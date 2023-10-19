import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from loguru import logger
from tqdm import tqdm
from PIL import Image
import csv
from skimage.metrics import structural_similarity
from omegaconf import OmegaConf
import datetime
import copy

from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid


import loralib as lora
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config,count_params
from ldm.modules.ema import LitEma
from loader import LSUN
from setting import options

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}

class DiffusionWrapper(nn.Module):
    def __init__(self, diff_model_config, conditioning_key=None):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            logger.error(f"diffusion model conditioning_key [{self.conditioning_key}] error")
            raise NotImplementedError()

        return out

class FuzzyDiffusion(pl.LightningModule):
    def __init__(self, diffusion_model, rule_num):
        super().__init__()
        self.conditioning_key = diffusion_model.conditioning_key # DiffusionWrapper
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']
        diffusion_models =[]
        for _ in range(rule_num):
            diffusion_models.append(copy.deepcopy(diffusion_model))
        self.diffusion_models = nn.ModuleList(diffusion_models)
        self.rule_num = rule_num
        self.fires = []
    
    def forward(self,  x, t, c_concat: list = None, c_crossattn: list = None):
        x_fuzz = []
        for i in range(self.rule_num):
            diffusion_model = self.diffusion_models[i]
            x_fuzz.append(diffusion_model(x, t, c_concat, c_crossattn))
        x_recon = 0
        for i in range(self.rule_num):
            x_recon = x_recon + self.fires[i]*x_fuzz[i]
        return x_recon
    
class FuzzyLatentDiffusion(LatentDiffusion):
    def __init__(self, rule_num, delegates_path,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rule_num = rule_num
        unet = self.model.diffusion_model
        lora.mark_only_lora_as_trainable(unet) # lora layers trainable
        fdm = FuzzyDiffusion(self.model, rule_num)# fuzzy diffusion model
        self.model = fdm
        count_params(self.model, verbose=True)
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")
        self.delegates = self.load_delegates(delegates_path)
        self.fires = [] # membership
        self.to_img = transforms.ToPILImage()
        
    def load_delegates(self, delegate_path):
        # load delegates of fuzzy system rule antecedent
        delegates = []
        with open(delegate_path, encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            # print(headers)
            for row in reader:
                img_path = row[0]
                img = Image.open(img_path)
                img = np.array(img)
                delegates.append(img)
        return delegates
        
    def membership(self, x, delegate):
        h,w,c = x.size() #numpy => h,w,c ; tensor => c,h,w
        x = x.view(c,h,w)
        im1 = self.to_img(x).convert('L') # gray
        im2 = Image.fromarray(delegate).convert('L') # gray
        im1 = np.uint8(im1)
        im2 = np.uint8(im2)
        latent_ssim_score = structural_similarity(im1, im2)
        return latent_ssim_score
    
    def antecedent(self, batch):
        # membership array
        batch = batch['image']
        batch_size = len(batch)
        u_arr = np.empty((self.rule_num),dtype=float)
        for i in range(self.rule_num):
            membership = 0
            for j in range(batch_size):
                membership = membership + self.membership(batch[j], self.delegates[i])
            u_arr[i] = membership / batch_size
        # normalization
        max = np.max(u_arr,axis=0)
        fires = u_arr/max
        return fires
    
    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.fires = self.antecedent(batch)
        self.model.fires = self.fires
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)

    def forward(self, x, c, *args, **kwargs):
        return super().forward(x, c, *args, **kwargs)

def config_learning_rate(model, model_config):
    # configure learning rate
    base_lr = model_config.base_learning_rate
    model.learning_rate = base_lr
    logger.info("++++ NOT USING LR SCALING ++++")
    logger.info(f"Setting learning rate to {model.learning_rate:.2e}")

if __name__ == '__main__':
    lsun_csv_path = "/home/yang/sda/github/fuzzydiffusion/output/img/lsun/lsun_churches.csv"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path+'output/log/'+now +'/fuzzy-latent-diffusion/'
    ckpt_path = root_path + 'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.ckpt'
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.log')
    config = OmegaConf.load("/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_churches-ldm-kl-8.yaml")
    model_config = config.pop("model", OmegaConf.create())
    # lightning_config = config.pop("lightning", OmegaConf.create())
    # trainer_config = lightning_config.pop("trainer", OmegaConf.create())
    # data_config = config.pop("data", OmegaConf.create())
    # unet_config = model_config.params.unet_config
    # first_stage_config =  model_config.params.first_stage_config
    # cond_stage_config = model_config.params.cond_stage_config
    # first_stage_model = instantiate_from_config(first_stage_config).cpu()
    # cond_stage_model = instantiate_from_config(cond_stage_config).cpu()
    fld_model = instantiate_from_config(model_config).to(options.device)
    config_learning_rate(fld_model, model_config)
    tb_logger = TensorBoardLogger(save_dir=root_path, name='tensor_board')
    csv_logger = CSVLogger(save_dir=root_path, name='csv_logs', flush_logs_every_n_steps=1)
    trainer = pl.Trainer(accelerator="gpu", #gpu
                        devices=1, 
                        max_epochs=20, #20
                        logger=[tb_logger, csv_logger],
                        log_every_n_steps=1,
                        benchmark=True, 
                        # max_steps= 100,
                        default_root_dir=root_path)
    dataset = LSUN('lsun', 'churches','train', transform=transforms.Compose([
            transforms.Resize((256,256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    train_data = DataLoader(
        dataset, batch_size=32, num_workers=5,shuffle=True, drop_last=True, pin_memory=True)
    trainer.fit(fld_model, train_data)
    trainer.save_checkpoint(ckpt_path)
    logger.remove(log_file)
            
    