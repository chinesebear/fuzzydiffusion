import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from loguru import logger
from tqdm import tqdm
from PIL import Image
import csv
from torchvision.transforms import ToTensor,ToPILImage
from skimage.metrics import structural_similarity
from omegaconf import OmegaConf
import datetime

from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning.utilities.distributed import rank_zero_only

from util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ema import LitEma
from distributions import normal_kl, DiagonalGaussianDistribution
from autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from util import make_beta_schedule, extract_into_tensor, noise_like
from ddim import DDIMSampler
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
    # Fuzzy Diffusion based on classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 rule_num = 2,
                 delegate_path = '',
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        logger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.rule_num = rule_num
        self.fires = [] # membership
        self.delegates = self.load_delegates(delegate_path) 
        self.models = nn.ModuleList([DiffusionWrapper(unet_config, conditioning_key) for i in range(self.rule_num)])
        self.model_ema = []
        for i in range(rule_num):
            count_params(self.models[i], verbose=True)
            self.use_ema = use_ema
            if self.use_ema:
                self.model_ema.append(LitEma(self.models[i]))
                logger.info(f"Keeping EMAs of {len(list(self.model_ema[i].buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

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
    
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()
    
    def forward(self, x, *args, **kwargs):
        return

# class FuzzyLatentDiffusion(FuzzyDiffusion):
#     """main class"""
#     def __init__(self,
#                  first_stage_config,
#                  cond_stage_config,
#                  num_timesteps_cond=None,
#                  cond_stage_key="image",
#                  cond_stage_trainable=False,
#                  concat_mode=True,
#                  cond_stage_forward=None,
#                  conditioning_key=None,
#                  scale_factor=1.0,
#                  scale_by_std=False,
#                  *args, **kwargs):
#         self.num_timesteps_cond = default(num_timesteps_cond, 1)
#         self.scale_by_std = scale_by_std
#         assert self.num_timesteps_cond <= kwargs['timesteps']
#         # for backwards compatibility after implementation of DiffusionWrapper
#         if conditioning_key is None:
#             conditioning_key = 'concat' if concat_mode else 'crossattn'
#         if cond_stage_config == '__is_unconditional__':
#             conditioning_key = None
#         ckpt_path = kwargs.pop("ckpt_path", None)
#         ignore_keys = kwargs.pop("ignore_keys", [])
#         super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
#         self.concat_mode = concat_mode
#         self.cond_stage_trainable = cond_stage_trainable
#         self.cond_stage_key = cond_stage_key
#         try:
#             self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
#         except:
#             self.num_downs = 0
#         if not scale_by_std:
#             self.scale_factor = scale_factor
#         else:
#             self.register_buffer('scale_factor', torch.tensor(scale_factor))
#         self.instantiate_first_stage(first_stage_config)
#         self.instantiate_cond_stage(cond_stage_config)
#         self.cond_stage_forward = cond_stage_forward
#         self.clip_denoised = False
#         self.bbox_tokenizer = None  

#         self.restarted_from_ckpt = False
#         if ckpt_path is not None:
#             self.init_from_ckpt(ckpt_path, ignore_keys)
#             self.restarted_from_ckpt = True

#         self.ToImg = ToPILImage()

#     def membership(self, x, delegate):
#         im1 = np.array(self.ToImg(x))
#         im2 = delegate
#         im1 = np.uint8(im1)
#         im2 = np.uint8(im2)
#         (latent_ssim_score, diff) = structural_similarity(x, delegate, win_size=101, channel_axis=0, full=True)
#         return latent_ssim_score
    
#     def antecedent(self, batch):
#         # membership array
#         batch_size = len(batch)
#         u_arr = np.empty((self.rule_num),dtype=float)
#         for i in range(self.rule_num):
#             membership = 0
#             for j in range(batch_size):
#                 membership = self.membership(batch[j], self.delegates[i])
#             u_arr[i] = membership / batch_size
#         # normalization
#         max = np.max(u_arr,axis=0)
#         fires = u_arr/max
#         self.fires = fires
#         return fires
    
#     def instantiate_first_stage(self, config):
#         model = instantiate_from_config(config)
#         self.first_stage_model = model.eval()
#         for param in self.first_stage_model.parameters():
#             param.requires_grad = False

#     def instantiate_cond_stage(self, config):
#         model = instantiate_from_config(config)
#         self.cond_stage_model = model.eval()
#         for param in self.cond_stage_model.parameters():
#             param.requires_grad = False
    
#     def get_input(self, batch, k):
#         x = batch[k]
#         if len(x.shape) == 3:
#             x = x[..., None]
#         x = rearrange(x, 'b h w c -> b c h w')
#         x = x.to(memory_format=torch.contiguous_format).float()
#         return x

#     def shared_step(self, batch):
#         x = self.get_input(batch, self.first_stage_key)
#         loss, loss_dict = self(x)
#         return loss, loss_dict
    
#     @rank_zero_only
#     @torch.no_grad()
#     def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
#         self.antecedent(batch)
        
#     @torch.no_grad()
#     def training_step(self, batch, batch_idx):
#         loss, loss_dict = self.shared_step(batch)

#         self.log_dict(loss_dict, prog_bar=False,
#                       logger=True, on_step=True, on_epoch=True)

#         self.log("global_step", self.global_step,
#                  prog_bar=False, logger=True, on_step=True, on_epoch=False)

#         if self.use_scheduler:
#             lr = self.optimizers().param_groups[0]['lr']
#             self.log('lr_abs', lr, prog_bar=False, logger=True, on_step=True, on_epoch=False)

#         return loss

#     @torch.no_grad()
#     def validation_step(self, batch, batch_idx):
#         _, loss_dict_no_ema = self.shared_step(batch)
#         with self.ema_scope():
#             _, loss_dict_ema = self.shared_step(batch)
#             loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
#         self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
#         self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
#     @torch.no_grad()
#     def test_step(self, batch, batch_idx):
#         _, loss_dict_no_ema = self.shared_step(batch)
#         with self.ema_scope():
#             _, loss_dict_ema = self.shared_step(batch)
#             loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
#         self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
#         self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

#     def on_train_batch_end(self, *args, **kwargs):
#         for i in range(self.rule_num):
#             if self.use_ema:
#                 self.model_ema(self.models[i])

#     def configure_optimizers(self):
#         params=[]
#         for i in range(self.rule_num):
#             lr = self.learning_rate
#             if len(params) ==0:
#                 params = list(self.models[i].parameters())
#             else:
#                 params = params + list(self.models[i].parameters())
#             if self.learn_logvar:
#                 params = params + [self.logvar]
#         opt = torch.optim.AdamW(params, lr=lr)
#         return opt

#     def forward(self, x, c, *args, **kwargs):
#         t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
#         if self.model.conditioning_key is not None:
#             assert c is not None
#             if self.cond_stage_trainable:
#                 c = self.get_learned_conditioning(c)
#             if self.shorten_cond_schedule:  # TODO: drop this option
#                 tc = self.cond_ids[t].to(self.device)
#                 c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
#         return self.p_losses(x, c, t, *args, **kwargs)
    
    
class FuzzyLatentDiffusion(L.LightningModule):
    def __init__(self, first_stage_model, cond_stage_model, model_config):
        super(FuzzyLatentDiffusion, self).__init__()
        self.first_stage_model = first_stage_model
        self.cond_stage_model = cond_stage_model
        self.unet_config = model_config.params.unet_config
        self.num_timesteps = 1000
        self.rule_num = 3
        self.delegates_path = model_config.params.delegate_path
        self.delegates = self.load_delegates(self.delegates_path)
        self.learning_rate = 1e-3
        self.diffusion_model = nn.ModuleList([DiffusionWrapper(self.unet_config).to(options.device) for _ in range(self.rule_num)])
        self.ToImg = transforms.ToPILImage()
        self.scale_factor = 1.0
        self.beta_1 = 0.0015
        self.beta_T = 0.0155
        self.T = self.num_timesteps
        self.register_buffer('betas', torch.linspace(self.beta_1, self.beta_T, self.T).float())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        # calculations for diffusion q(x_t | x_{t-1}) and others
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
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
        im1 = self.ToImg(x).convert('L') # gray
        im2 = Image.fromarray(delegate).convert('L') # gray
        im1 = np.uint8(im1)
        im2 = np.uint8(im2)
        latent_ssim_score = structural_similarity(im1, im2)
        return latent_ssim_score
    
    def antecedent(self, batch):
        # membership array
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
        self.fires = fires
        return fires
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
    
    def img_generate(self, imgs):
        b = len(imgs)
        h,w = imgs[0].size
        result = Image.new('RGB', (b * w, h))
        for i in range(b):
            result.paste(imgs[i], (i*w, 0))
        return result
    
    def training_step(self, batch, batch_idx):
        # batch
        img_batch = batch[0]
        text_batch = batch[1]
        x = img_batch # b c h w
        membership = self.antecedent(img_batch)
        z = self.first_stage_model.encode(x).sample() * self.scale_factor
        # t = torch.randint(0, self.num_timesteps, (x.shape[0],)).long().to(options.device)
        # noise = torch.randn_like(z) # noise
        # z_noisy = self.q_sample(x_start=z, t=t, noise=noise)
        # z_fuzz = []
        # for i in range(self.rule_num):
        #     diff_model = self.diffusion_model[i]
        #     z_fuzz.append(diff_model(z_noisy, t))
        # z_recon = 0
        # for i in range(self.rule_num):
        #     z_recon = z_recon + membership[i]*z_fuzz[i]
        x_recon = self.first_stage_model.decode(z_recon)
        loss = torch.nn.functional.mse_loss(x, x_recon)
        self.log('loss',loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=2)
        global_step = self.trainer.global_step
        if global_step% 10 == 0:
            img = self.img_generate([self.ToImg(x[0]),
                                    self.ToImg(x_recon[0]),
                                    self.ToImg(x[1]),
                                    self.ToImg(x_recon[1]),])
            img.save(self.trainer.default_root_dir+"/train.jpg")
        return loss
    
    def on_train_epoch_end(self):
        root_path = self.trainer.default_root_dir
        ckpt_path = root_path+"/ckpt/"
        epoch = self.trainer.current_epoch
        self.trainer.save_checkpoint(ckpt_path+"epoch"+str(epoch)+".ckpt")
        return
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        for i in range(self.rule_num):
            params = params + list(self.diffusion_model[i].parameters())
        opt = torch.optim.Adam(params, lr=lr)
        return opt


if __name__ == '__main__':
    
    delegates = []
    lsun_csv_path = "/home/yang/sda/github/fuzzydiffusion/output/img/lsun/lsun_churches.csv"
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path+'output/log/'+now +'/fuzzy-latent-diffusion/'
    ckpt_path = root_path + 'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.ckpt'
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.log')
    toTensor = ToTensor()
    with open(lsun_csv_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        # print(headers)
        for row in reader:
            path = row[0]
            img = Image.open(path)
            img_tensor= toTensor(img)
            delegates.append(img_tensor)
    config = OmegaConf.load("/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_churches-fuzzy-ldm-kl-8.yaml")
    model_config = config.pop("model", OmegaConf.create())
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = lightning_config.pop("trainer", OmegaConf.create())
    data_config = config.pop("data", OmegaConf.create())
    unet_config = model_config.params.unet_config
    first_stage_config =  model_config.params.first_stage_config
    cond_stage_config = model_config.params.cond_stage_config
    first_stage_model = instantiate_from_config(first_stage_config).cpu()
    cond_stage_model = instantiate_from_config(cond_stage_config).cpu()
    fld_model = FuzzyLatentDiffusion(first_stage_model, cond_stage_model, model_config).to(options.device)
    tb_logger = TensorBoardLogger(save_dir=root_path, name='tensor_board')
    csv_logger = CSVLogger(save_dir=root_path, name='csv_logs', flush_logs_every_n_steps=1)
    trainer = L.Trainer(accelerator="gpu", #gpu
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
        dataset, batch_size=2, shuffle=True, drop_last=True, pin_memory=True)
    trainer.fit(fld_model, train_data)
    trainer.save_checkpoint(ckpt_path)
    logger.remove(log_file)
            
    