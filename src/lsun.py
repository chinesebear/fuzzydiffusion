import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
from loguru import logger
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config,count_params
from ldm.modules.ema import LitEma
from loader import LSUN
from setting import options
from metrics import Evaluator
from fldm import FuzzyLatentDiffusion,load_delegates
from utils import setup_seed,load_model,save_model,frozen_model,\
    activate_model,stat_model_param,csv_dist_record,metrics_mean,\
    combine_imgs3,combine_imgs2,csv_record,metrics_fig,img_norm,\
    combine_imgs4,GradualWarmupScheduler,check_dir



def lsun_train(dataset_name, config_path, delegate_path, rule_num):
    setup_seed(10)
    batch_size = 32
    img_shape = (3,256,256)
    z_shape = (4,32,32)
    epoch = 20
    beta_1 = 0.0015 # 1e-4
    beta_T= 0.0155 # 0.02
    T = 1000
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path+f'output/log/lsun_{dataset_name}/'+now +'/'
    model_path = options.base_path+ f"output/pth/"
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-train-'+str(datetime.date.today()) +'.log')
    check_dir(root_path)
    check_dir(model_path)
    check_dir(log_file)
    
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    ldmodel = instantiate_from_config(model_config).to(options.device)
    delegates = load_delegates(delegate_path)
    fldmodel = FuzzyLatentDiffusion(ldmodel,
                                    rule_num,
                                    img_shape,
                                    z_shape,
                                    delegates,
                                    beta_1,
                                    beta_T,
                                    T,
                                    root_path).to(options.device)
    frozen_model(fldmodel)
    stat_model_param(fldmodel, "fldmodel")
    
    dataset = LSUN('/home/yang/sda/github/fuzzydiffusion/output/datasets/lsun', 
                   dataset_sub_path=dataset_name,
                   phase='train', 
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #https://blog.csdn.net/zylooooooooong/article/details/122805833
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    train_data = DataLoader(
        dataset, batch_size=batch_size, num_workers=5,shuffle=True, drop_last=True, pin_memory=True)
    
    if not os.path.exists(model_path+f"fldm_lsun_{dataset_name}_epoch_{epoch}_rules_{rule_num}.pt"):
        fldmodel.fuzzy_trainer(train_data, epoch)
        save_model(fldmodel, model_path+f"fldm_lsun_{dataset_name}_epoch_{epoch}_rules_{rule_num}.pt")

    logger.info("lsun test start")
    lsun_test(dataset_name, fldmodel, root_path)
    logger.info("lsun test done")
    logger.remove(log_file)

def lsun_test(dataset_name, fldmodel, root_path):
    fldmodel.eval()
    frozen_model(fldmodel)
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-test-'+str(datetime.date.today()) +'.log')
    metrics_csv_path = root_path+f'lsun_{dataset_name}_{fldmodel.rule_num}rules_metrics-'+str(datetime.date.today()) +'.csv'
    dist_csv_path = root_path+'distribution-'+str(datetime.date.today()) +'.csv'
    
    eval = Evaluator(torch.device("cpu"))
    dataset = LSUN('/home/yang/sda/github/fuzzydiffusion/output/datasets/lsun', 
                   dataset_sub_path=dataset_name,
                   phase='train', 
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

    for i in tqdm(range(200)): # 200*32 = 6400
        batch = next(iter(test_data))
        x = batch['image'].cuda()
        x, z, z_fuzz, x_fuzz, fire = fldmodel(batch,latent_output=True)
        
        x_p = fldmodel.ldmodel.decode_first_stage(z)
        
        # compress [-1,1]  into  [0,1]
        x_norm = sigmoid(x)
        x_p_norm = sigmoid(x_p)
        x_fuzz_norm =  sigmoid(x_fuzz)
        
        
        distributions = {}
        distributions['x'] = x
        distributions['z'] = z
        distributions['x_p'] = x_p
        distributions['x_fuzz'] = x_fuzz
        distributions['x_norm'] = x_norm
        distributions['x_p_norm'] = x_p_norm
        distributions['x_fuzz_norm'] = x_fuzz_norm
        csv_dist_record(dist_csv_path, distributions)
        combine_imgs3(x_norm, x_p_norm,x_fuzz_norm)
        
        real_imgs = x_p_norm.cpu()
        gen_imgs = x_fuzz_norm.cpu()
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
    lsun_train('churches', 
               f"{options.base_path}src/config/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
               f"{options.base_path}output/delegates/lsun_church/lsun_church_3_delegates.csv",
               3)
    lsun_train('bedrooms', 
               f"{options.base_path}src/config/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml",
               f"{options.base_path}output/delegates/lsun_bedroom/lsun_bedroom_3_delegates.csv",
               3)