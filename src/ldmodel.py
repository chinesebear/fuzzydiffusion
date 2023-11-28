import datetime
import copy
import os

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


import loralib as lora
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config,count_params
from ldm.modules.ema import LitEma
from loader import LSUN
from setting import options,setup_seed
from metrics import Evaluator

def tensor_to_img(img_tensors):
    imgs = []
    toImg = transforms.ToPILImage()
    for tensor in img_tensors:
        img = toImg(tensor)
        imgs.append(img)
    return imgs

def combine_imgs(x, x_p, x_rec):
    b,c,h,w = x.shape
    
    x_imgs = tensor_to_img(x)
    x_p_imgs = tensor_to_img(x_p)
    x_rec_imgs = tensor_to_img(x_rec)
 
    # 创建空白长图
    result = Image.new("RGB", (w*3, h*b))
 
    # 拼接图片    
    for i in range(b):
        result.paste(x_imgs[i], box=(0, i*h))
        result.paste(x_p_imgs[i], box=(w, i*h))
        result.paste(x_rec_imgs[i], box=(w*2, i*h))
 
    # 保存图片
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/lsun.jpg")

def csv_record(path, data):
    header = ['fid', 'is', 'mifid', 'kid',
              'psnr','ms_ssim','ssim',
              'precision', 'recall']  
    if len(data) != len(header):
        logger.error("csv header field number error")
        return
    if os.path.exists(path):
        with open(path, 'a',newline='') as f:
            write = csv.writer(f)
            write.writerow(data)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(data)

def csv_dist_record(path, data):
    header = ['x', 'z', 'z_noisy', 'z_rec',
              'x_rec','x_p','x_norm', "x_p_norm","x_rec_norm"]
    if len(data) != len(header):
        logger.error("csv header field number error")
        return
    if os.path.exists(path):
        with open(path, 'a',newline='') as f:
            write = csv.writer(f)
            row = []
            for d in data:
                row.append(f"{round(d.mean().item(),2)}/{round(d.std().item(),2)}")
            write.writerow(row)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            row = []
            for d in data:
                row.append(f"{round(d.mean().item(),2)}/{round(d.std().item(),2)}")
            write.writerow(row)

def lsun_test(dataset_name, config_path):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path+f'output/log/lsun_{dataset_name}/'+now +'/'
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.log')
    csv_path = root_path+'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.csv'
    dist_csv_path = root_path+'fuzzy-latent-diffusion-dist-'+str(datetime.date.today()) +'.csv'
    
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    ldmodel = instantiate_from_config(model_config).to(options.device)

    eval = Evaluator(device=torch.device("cpu"))
    dataset = LSUN('lsun', dataset_name,'train', data_limit= 6400,
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #https://blog.csdn.net/zylooooooooong/article/details/122805833
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    batch_size = 32
    test_data = DataLoader(
        dataset, batch_size=batch_size, num_workers=5,shuffle=True, drop_last=True, pin_memory=True)
    
    real_imgs = torch.empty((test_data.dataset.data_len, 3, 256, 256)).cpu()
    gen_imgs = torch.empty((test_data.dataset.data_len, 3, 256, 256)).cpu()
    r_offset = 0
    g_offset = 0
    for batch in tqdm(test_data):
        bs = len(batch)
        x = batch['image'].cuda()
        z = ldmodel.get_input(batch, 'image')[0]

        z_start = z
        t = torch.randint(0, ldmodel.num_timesteps, (z_start.shape[0],), device=ldmodel.device).long()
        t = torch.ones_like(t).long()*(ldmodel.num_timesteps -1)
        noise = torch.randn_like(z_start)
        z_noisy = ldmodel.q_sample(x_start=z_start, t=t, noise=noise)
        ddim = DDIMSampler(ldmodel)
        shape = x.shape
        bs = shape[0]
        shape = shape[1:]
        samples, _ = ddim.sample(200, batch_size=bs, shape=shape, eta=0.0, verbose=False,x0=z_start, x_T=z_noisy)
        # samples = ldmodel.sample(cond=None, batch_size=batch_size, x_T=z_noisy)
        z_rec = samples
        x_rec = ldmodel.decode_first_stage(z_rec)
        x_p = ldmodel.decode_first_stage(z)
        
        x_rec_norm =  x_rec/2 + 0.5# [-1,1]  -> [0,1]
        x_p_norm = x_p/2 + 0.5
        x_norm = x/2 + 0.5
        
        combine_imgs(x_norm, x_p_norm,x_rec_norm)
        
        for i in range(bs):
            real_imgs[r_offset+i] = x_norm[i].cpu()
            gen_imgs[g_offset+i] = x_rec_norm[i].cpu()
        r_offset = r_offset + bs
        g_offset = g_offset + bs
        
    fid = eval.calc_fid(real_imgs, gen_imgs)
    _is = eval.calc_is(gen_imgs)
    mifid = eval.calc_mifid(real_imgs, gen_imgs)
    kid = eval.calc_kid(real_imgs, gen_imgs)
    logger.info(f"fid:{fid},is:{_is},mifid:{mifid},kid:{kid}")
    
    pnsr = eval.calc_psnr(real_imgs, gen_imgs)
    ms_ssim = eval.calc_ms_ssim(real_imgs, gen_imgs)
    ssim = eval.calc_ssim(real_imgs, gen_imgs)
    logger.info(f"psnr:{pnsr},ms_ssim:{ms_ssim},ssim:{ssim}")
    
    precision,recall = eval.calc_preision_recall(real_imgs, gen_imgs)
    logger.info(f"precision:{precision},recall:{recall}")
    
    # total_precision = 0.0
    # total_recall = 0.0
    # total_psnr = 0.0
    # total_ms_ssim = 0.0
    # total_ssim = 0.0
    # count = 0
    # for batch in tqdm(test_data):
    #     x = batch['image'].cuda()
    #     z = ldmodel.get_input(batch, 'image')[0]
    #     z_start = z
    #     t = torch.randint(0, ldmodel.num_timesteps, (z_start.shape[0],), device=ldmodel.device).long()
    #     t = torch.ones_like(t).long()*(ldmodel.num_timesteps -1)
    #     noise = torch.randn_like(z_start)
    #     z_noisy = ldmodel.q_sample(x_start=z_start, t=t, noise=noise)
    #     ddim = DDIMSampler(ldmodel)
    #     shape = x.shape
    #     bs = shape[0]
    #     shape = shape[1:]
    #     samples, _ = ddim.sample(200, batch_size=bs, shape=shape, eta=0.0, verbose=False,x0=z_start, x_T=z_noisy)
    #     # samples = ldmodel.sample(cond=None, batch_size=batch_size, x_T=z_noisy)
    #     z_rec = samples
    #     x_rec = ldmodel.decode_first_stage(z_rec)
    #     x_p = ldmodel.decode_first_stage(z)
        
    #     # x_rec_norm = (x_rec-x_rec.mean())/x_rec.std()/2 + 0.5 # [-1,1]  -> [0,1]
    #     # x_p_norm = (x_p-x_p.mean())/x_p.std()/2 + 0.5
    #     # x_norm = (x-x.mean())/x.std()/2 + 0.5
    #     x_rec_norm =  x_rec/2 + 0.5# [-1,1]  -> [0,1]
    #     x_p_norm = x_p/2 + 0.5
    #     x_norm = x/2 + 0.5
        
    #     csv_dist_record(dist_csv_path, [x,z,z_noisy,z_rec, x_rec,x_p, x_norm, x_p_norm, x_rec_norm])
    #     combine_imgs(x_norm, x_p_norm,x_rec_norm)
        
    #     count = count + 1
    #     real_imgs = x_norm
    #     gen_imgs = x_rec_norm
    #     fid = eval.calc_fid(real_imgs, gen_imgs)
    #     _is = eval.calc_is(gen_imgs)
    #     mifid = eval.calc_mifid(real_imgs, gen_imgs)
    #     kid = eval.calc_kid(real_imgs, gen_imgs)
    #     logger.info(f"fid:{fid},is:{_is},mifid:{mifid},kid:{kid}")
        
    #     pnsr = eval.calc_psnr(real_imgs, gen_imgs)
    #     ms_ssim = eval.calc_ms_ssim(real_imgs, gen_imgs)
    #     ssim = eval.calc_ssim(real_imgs, gen_imgs)
    #     total_psnr = total_psnr + pnsr
    #     total_ms_ssim = total_ms_ssim + ms_ssim
    #     total_ssim = total_ssim + ssim
    #     logger.info(f"psnr:{round(total_psnr/count,2)},ms_ssim:{round(total_ms_ssim/count,2)},ssim:{round(total_ssim/count,2)}")
        
    #     precision,recall = eval.calc_preision_recall(real_imgs, gen_imgs)
    #     total_precision = total_precision+ precision
    #     total_recall = total_recall + recall
    #     logger.info(f"precision:{round(total_precision/count,2)},recall:{round(total_recall/count,2)}")
        
    #     record_row = [str(fid), 
    #                   str(_is), 
    #                   str(mifid), 
    #                   str(kid), 
    #                   str(round(total_psnr/count,2)),
    #                   str(round(total_ms_ssim/count,2)),
    #                   str(round(total_ssim/count,2)),
    #                   str(round(total_precision/count,2)),
    #                   str(round(total_recall/count,2))]
    #     csv_record(csv_path, record_row)
    # logger.remove(log_file)

if __name__ == '__main__':
    setup_seed(10)
    #  lsun_test('churches', "/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_churches-ldm-kl-8.yaml")
    lsun_test('bedrooms', "/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml")