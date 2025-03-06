import datetime
import copy
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
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
    
def metrics_fig(name, data):
    xmax = len(data)
    xpoints = np.arange(xmax)
    ypoints = np.array(data)
    
    plt.figure(figsize=(5,5)) 
    plt.title(name)
    plt.plot(xpoints, ypoints, color='blue')
    plt.xlabel("batch")
    plt.ylabel(f"{name} score")
    plt.savefig(f"/home/yang/sda/github/fuzzydiffusion/output/img/metrics_{name}.jpg")
    plt.show() 

def metrics_mean(data):
    count = len(data)
    sum = 0.0
    for i in range(count):
        sum = sum + data[i]
    mean = sum / count
    return round(mean, 2)

def csv_record(path, data):
    header = ['fid', 'is', 'mifid', 'kid',
              'psnr','ms_ssim','ssim',
              'precision', 'recall']  
    row = []
    for name in header:
        row.append(data[name])
    if os.path.exists(path):
        with open(path, 'a',newline='') as f:
            write = csv.writer(f)
            write.writerow(row)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)

def csv_dist_record(path, data):
    header = ['x', 'z', 'z_noisy', 'z_rec',
              'x_rec','x_p','x_norm', "x_p_norm","x_rec_norm"]
    row_data = []
    for name in header:
        row_data.append(data[name])
    if os.path.exists(path):
        with open(path, 'a',newline='') as f:
            write = csv.writer(f)
            row = []
            for d in row_data:
                row.append(f"{round(d.mean().item(),2)}/{round(d.std().item(),2)}")
            write.writerow(row)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            row = []
            for d in row_data:
                row.append(f"{round(d.mean().item(),2)}/{round(d.std().item(),2)}")
            write.writerow(row)

def lsun_test(dataset_name, config_path):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path+f'output/log/lsun_{dataset_name}/'+now +'/'
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-'+str(datetime.date.today()) +'.log')
    metrics_csv_path = root_path+'metrics-'+str(datetime.date.today()) +'.csv'
    dist_csv_path = root_path+'distribution-'+str(datetime.date.today()) +'.csv'
    
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    ldmodel = instantiate_from_config(model_config).to(options.device)
    ddim = DDIMSampler(ldmodel)

    # eval = Evaluator(device=torch.device("cpu"))
    eval = Evaluator()
    dataset = LSUN('/home/yang/sda/github/fuzzydiffusion/output/datasets/lsun', 
                   dataset_sub_path=dataset_name,
                   phase='train', 
                   data_limit= 6400,
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
    
    sigmoid = nn.Sigmoid()
    met_data = {}
    met_data['fid'] = []
    met_data['is'] = []
    met_data['mifid'] = []
    met_data['kid'] = []
    met_data['psnr'] = []
    met_data['ms_ssim'] = []
    met_data['ssim'] = []
    met_data['precision'] = []
    met_data['recall'] = []
    met_data['psnr_m'] = []
    met_data['ms_ssim_m'] = []
    met_data['ssim_m'] = []
    met_data['precision_m'] = []
    met_data['recall_m'] = []
    
    count = 0
    for batch in tqdm(test_data):
        x = batch['image'].cuda()
        z = ldmodel.get_input(batch, 'image')[0]
        z_start = z
        t = torch.randint(0, ldmodel.num_timesteps, (z_start.shape[0],), device=ldmodel.device).long()
        t = torch.ones_like(t).long()*(ldmodel.num_timesteps -1)
        noise = torch.randn_like(z_start)
        z_noisy = ldmodel.q_sample(x_start=z_start, t=t, noise=noise)
        
        bs,c,h,w = z.shape
        shape = (c,h,w)
        samples, _ = ddim.sample(200, batch_size=bs, shape=shape, eta=0.0, verbose=False,x0=z_start, x_T=z_noisy)
        # samples = ldmodel.sample(cond=None, batch_size=batch_size, x_T=z_noisy)
        z_rec = samples
        x_rec = ldmodel.decode_first_stage(z_rec)
        x_p = ldmodel.decode_first_stage(z)
        
        # compress [-1,1]  into  [0,1]
        x_rec_norm =  sigmoid(x_rec)
        x_p_norm = sigmoid(x_p)
        x_norm = sigmoid(x)
        
        distributions = {}
        distributions['x'] = x
        distributions['z'] = z
        distributions['z_noisy'] = z_noisy
        distributions['z_rec'] = z_rec
        distributions['x_rec'] = x_rec
        distributions['x_p'] = x_p
        distributions['x_norm'] = x_norm
        distributions['x_p_norm'] = x_p_norm
        distributions['x_rec_norm'] = x_rec_norm
        csv_dist_record(dist_csv_path, distributions)
        combine_imgs(x_norm, x_p_norm,x_rec_norm)
        
        count = count + 1
        real_imgs = x_norm
        gen_imgs = x_rec_norm
        fid = eval.calc_fid(real_imgs, gen_imgs)
        _is = eval.calc_is(gen_imgs)
        mifid = eval.calc_mifid(real_imgs, gen_imgs)
        kid = eval.calc_kid(real_imgs, gen_imgs)
        met_data['fid'].append(fid)
        met_data['is'].append(_is)
        met_data['mifid'].append(mifid)
        met_data['kid'].append(kid)
        logger.info(f"fid:{fid},is:{_is},mifid:{mifid},kid:{kid}")
        
        psnr = eval.calc_psnr(real_imgs, gen_imgs)
        ms_ssim = eval.calc_ms_ssim(real_imgs, gen_imgs)
        ssim = eval.calc_ssim(real_imgs, gen_imgs)
        met_data['psnr'].append(psnr)
        met_data['ms_ssim'].append(ms_ssim)
        met_data['ssim'].append(ssim)
        logger.info(f"psnr:{metrics_mean(met_data['psnr'])},ms_ssim:{metrics_mean(met_data['ms_ssim'])},ssim:{metrics_mean(met_data['ssim'])}")
        
        precision,recall = eval.calc_preision_recall(real_imgs, gen_imgs)
        met_data['precision'].append(precision)
        met_data['recall'].append(recall)
        logger.info(f"precision:{metrics_mean(met_data['precision'])},recall:{metrics_mean(met_data['recall'])}")
        
        met_data['psnr_m'].append(metrics_mean(met_data['psnr']))
        met_data['ms_ssim_m'].append(metrics_mean(met_data['ms_ssim']))
        met_data['ssim_m'].append(metrics_mean(met_data['ssim']))
        met_data['precision_m'].append(metrics_mean(met_data['precision']))
        met_data['recall_m'].append(metrics_mean(met_data['recall']))
        
        record_row = {}
        record_row['fid'] = str(fid)
        record_row['is'] = str(_is)
        record_row['mifid'] = str(mifid)
        record_row['kid'] = str(kid)
        record_row['psnr'] = str(metrics_mean(met_data['psnr']))
        record_row['ms_ssim'] = str(metrics_mean(met_data['ms_ssim']))
        record_row['ssim'] = str(metrics_mean(met_data['ssim']))
        record_row['precision'] = str(metrics_mean(met_data['precision']))
        record_row['recall'] = str(metrics_mean(met_data['recall']))
        
        csv_record(metrics_csv_path, record_row)
        metrics_fig("fid", met_data['fid'])
        metrics_fig("precision", met_data['precision_m'])
        metrics_fig("recall", met_data['recall_m'])
        metrics_fig("psnr", met_data['psnr_m'])
    logger.remove(log_file)

if __name__ == '__main__':
    setup_seed(10)
    # lsun_test('churches', "/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_churches-ldm-kl-8.yaml")
    lsun_test('bedrooms', "/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml")