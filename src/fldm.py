import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np


from loguru import logger
from tqdm import tqdm
from PIL import Image
import csv
from skimage.metrics import structural_similarity
from omegaconf import OmegaConf

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config,count_params
from ldm.modules.ema import LitEma
from loader import LSUN
from setting import options
from metrics import Evaluator
from utils import frozen_model,activate_model,save_model,\
                csv_record,GradualWarmupScheduler,extract

transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

def load_delegates(delegate_path):
    # load delegates of fuzzy system rule antecedent
    delegates = []
    with open(delegate_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        logger.info(f"csv {header}")
        for row in reader:
            img_path = row[0]
            # text = row[1]
            img = Image.open(img_path)
            img = transform(img)
            delegates.append(img)
            # delegates.append([img,text])
    return delegates

class MLP(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super().__init__()
        hidden_size = input_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    def forward(self, input):
        input_data = input
        hidden_data = self.input_layer(input_data)
        hidden_data = self.hidden_layer(hidden_data)
        output_data = self.output_layer(hidden_data)
        return output_data
    
class conv2d_bn(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        if kernel_size == 3:
            self.conv = nn.Conv2d(4,4,kernel_size=kernel_size,stride=1,padding=1) 
        elif kernel_size == 1:
            self.conv = nn.Conv2d(4,4,kernel_size=kernel_size,stride=1) 
        self.bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.relu(output)
        return output


class Inception(nn.Module):
    def __init__(self, c, h, w):
        super().__init__()
        self.conv1=conv2d_bn(kernel_size=1)
        
        self.conv2a=conv2d_bn(kernel_size=1)
        self.conv2b=conv2d_bn(kernel_size=3)
        
        self.avg_pool3=nn.AvgPool2d(kernel_size=3,stride=1, padding=1)
        self.conv3=conv2d_bn(kernel_size=3)
        
        self.conv4a=conv2d_bn(kernel_size=1)
        self.conv4b=conv2d_bn(kernel_size=3)
        self.conv4c=conv2d_bn(kernel_size=3)
        
        input_size = h*w*4
        output_size = h*w
        # self.fc = nn.Linear(input_size, output_size)
        self.fc = MLP(input_size, output_size)
        
    def forward(self,x):
        x1=self.conv1(x)
        # print(f"x1.shape:{x1.shape}")
        
        x2=self.conv2a(x)
        x2=self.conv2b(x2)
        # print(f"x2.shape:{x2.shape}")
        
        x3=self.avg_pool3(x)
        x3= self.conv3(x3)
        # print(f"x3.shape:{x3.shape}")
        
        x4 = self.conv4a(x)
        x4 = self.conv4b(x4)
        x4 = self.conv4c(x4)
        # print(f"x4.shape:{x4.shape}")
        
        b, c, h, w = x.shape
        x1 = x1.view(b, c,-1)
        x2 = x2.view(b, c,-1)
        x3 = x3.view(b, c,-1)
        x4 = x4.view(b, c,-1)
        
        x_cat = torch.cat((x1,x2,x3,x4), dim = -1)
        output = self.fc(x_cat).view(b,c,h,w)
        return output

class FuzzyLatentDiffusion(nn.Module):
    def __init__(self, ldmodel, rule_num, img_shape, z_shape, delegates,beta_1, beta_T, T, root_path, condition=False) -> None:
        super().__init__()
        self.ldmodel = ldmodel
        frozen_model(ldmodel)
        self.rule_num = rule_num
        self.img_shape = img_shape
        self.z_shape = z_shape # img shape of latent space
        unet = ldmodel.model
        self.rule_models = nn.ModuleList([copy.deepcopy(unet) for _ in range(self.rule_num)])
        self.fires = [] # membership
        self.condition = condition
        self.delegates = delegates #[img, text]
        self.evaluator = Evaluator()
        # self.sigmoid = nn.Sigmoid()
        self.init_trainer(beta_1, beta_T, T)
        self.root_path = root_path
    
    def init_trainer(self, beta_1, beta_T, T):
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        
    def membership(self, x, delegate):
        # c,h,w
        latent_ssim_score = self.evaluator.calc_ssim2(x, delegate)
        return latent_ssim_score

    def antecedent(self, batch):
        # membership array
        batch_size = len(batch)
        u_arr = torch.zeros([batch_size, self.rule_num])
        for i in range(batch_size):
            for j in range(self.rule_num):
                delegate = self.delegates[j] ## image
                delegate = delegate.to(options.device)
                membership = self.membership(batch[i], delegate)
                u_arr[i][j] = membership
        fire = u_arr
        return fire
    
    def fuzzy_sampler(self, batch, latent_output=False):
        x = batch['image'].to(options.device)
        fire = self.antecedent(x)
        z = self.ldmodel.get_input(batch, 'image')[0]
        bs,c,h,w = z.shape
        t = torch.ones(bs).long()*(self.ldmodel.num_timesteps -1)
        t = t.to(options.device)
        
        z_start = z
        noise = torch.randn_like(z_start).to(options.device)
        z_noisy = self.ldmodel.q_sample(x_start=z_start, t=t, noise=noise)
        
        shape = (c,h,w)
        rule_output = torch.empty([self.rule_num, bs, c, h, w]).to(options.device)
        for r in range(self.rule_num):
            self.ldmodel.model = self.rule_models[r]
            ddim = DDIMSampler(self.ldmodel)
            samples, _ = ddim.sample(200, batch_size=bs, shape=shape, eta=0.0, verbose=False,x0=z_start, x_T=z_noisy)
            rule_output[r] = samples
        
        rule_idx = fire.argmax(dim=-1)
        z_fuzz = torch.empty([bs, c, h, w]).to(options.device)
        for i in range(bs):
            r = rule_idx[i].item()
            z_fuzz[i] = rule_output[r][i]
            
        x_fuzz = self.ldmodel.decode_first_stage(z_fuzz)
        
        if latent_output:
            return x, z, z_fuzz, x_fuzz, fire
        return x_fuzz
            
    def fuzzy_trainer(self, train_data, epoch):
        logger.info(f"rule_models epoch{epoch} rules{self.rule_num} train start")
        for r in range(self.rule_num):
            logger.info(f"rule{r+1} train start")
            consequent_model = self.rule_models[r]
            activate_model(consequent_model)
            optimizer = torch.optim.Adam(consequent_model.parameters(), lr=options.learning_rate, weight_decay=1e-4)
            cosineScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=0, last_epoch=-1)
            warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=2.0, warm_epoch=epoch, after_scheduler=cosineScheduler)
            for e in tqdm(range(epoch), f"rule{r+1} train"):
                batch_count = 0
                for batch in tqdm(train_data, f"epoch{e}"):
                    x = batch['image'].to(options.device)
                    fire = self.antecedent(x).to(options.device)
                    z = self.ldmodel.get_input(batch, 'image')[0]
                    z_rule = torch.zeros_like(z)
                    rule_idx = fire.argmax(dim=-1)
                    offset = 0
                    for i in range(len(x)):
                        # train rule consequent
                        if rule_idx[i].item() !=  r:
                            continue
                        z_rule[offset] = z[i]
                        offset = offset + 1
                    if offset == 0:
                        continue
                    
                    optimizer.zero_grad()
                    t = torch.randint(self.T, size=(z_rule.shape[0], ), device=z_rule.device)
                    noise = torch.randn_like(z_rule)
                    z_rule_t = (
                        extract(self.sqrt_alphas_bar, t, z.shape) * z +
                        extract(self.sqrt_one_minus_alphas_bar, t, z.shape) * noise)
                    loss = F.mse_loss(consequent_model(z_rule_t, t), noise, reduction='none').mean()
                    loss.backward()
                    grad_clip = 1.0
                    torch.nn.utils.clip_grad_norm_(consequent_model.parameters(), grad_clip)
                    optimizer.step()
                
                    batch_count = batch_count + 1
                    csv_record(self.root_path+f"loss_rule_{r+1}.csv", {'epoch': f"{e}",'batch': f"{batch_count}",'loss': f"{loss.item()}"})
                    # if batch_count % 10 == 0:
                    #     logger.info(f"rule:{r+1}, epoch:{e}, batch:{batch_count}, loss:{round(loss.item(), 2)}")
                warmUpScheduler.step()
            frozen_model(consequent_model)
            logger.info(f"rule_models rule{r+1} train done")
        logger.info(f"rule_models epoch{epoch} rules{self.rule_num} train done")
        # save_model(self.rule_models, self.root_path+f"epoch_{epoch}_rules_{self.rule_num}_.pt")
    
    def forward(self, batch, latent_output=False):
        ## x -> z -> z_fuzz -> x_fuzz
        ## x -> x_p -> x_fuzz
        x, z, z_fuzz, x_fuzz, fire = self.fuzzy_sampler(batch,latent_output)
        if latent_output:
            return x, z, z_fuzz, x_fuzz, fire
        else:
            return x_fuzz
        
if __name__ == '__main__':
    print("done")

    