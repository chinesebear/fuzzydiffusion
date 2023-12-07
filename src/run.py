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
    combine_imgs4 



def lsun_train(dataset_name, config_path):
    setup_seed(10)
    batch_size = 32
    sample_batch_limit = 300
    sample_total_limit = batch_size*sample_batch_limit
    preprocess_total_limit = None
    preprocess_epoch = 4
    img_shape = (3,256,256)
    z_shape = (4,32,32)
    rule_num = 3
    epoch = 20
    step = 10
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path+f'output/log/lsun_{dataset_name}/'+now +'/'
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-train-'+str(datetime.date.today()) +'.log')
    
    config = OmegaConf.load(config_path)
    model_config = config.pop("model", OmegaConf.create())
    ldmodel = instantiate_from_config(model_config).to(options.device)
    delegates = load_delegates("/home/yang/sda/github/fuzzydiffusion/output/img/lsun/lsun_churches.csv")
    fldmodel = FuzzyLatentDiffusion(ldmodel,rule_num,img_shape,z_shape,delegates).to(options.device)
    frozen_model(fldmodel)
    # for i in range(rule_num):
    #     activate_model(fldmodel.kbmodel[i])
    stat_model_param(fldmodel, "fldmodel")
    
    # kbmodel, preprocess
    logger.info("kbmodel preprocess start")
    dataset = LSUN('/home/yang/sda/github/fuzzydiffusion/output/datasets/lsun', 
                   dataset_sub_path=dataset_name,
                   phase='train', 
                   data_limit= preprocess_total_limit,
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #https://blog.csdn.net/zylooooooooong/article/details/122805833
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    
    train_data = DataLoader(
        dataset, batch_size=batch_size, num_workers=5,shuffle=True, drop_last=True, pin_memory=True)
    criterion = nn.MSELoss()
    for r in range(rule_num):
        consequent_model = fldmodel.kbmodel[r]
        optimizer = torch.optim.Adam(consequent_model.parameters(), lr=options.learning_rate, weight_decay=0)
        activate_model(consequent_model)
        for e in tqdm(range(preprocess_epoch), f"preprocess rule_{r+1}"):
            batch_count = 0
            for batch in tqdm(train_data):
                x = batch['image'].to(options.device)
                fire = fldmodel.antecedent(x).to(options.device)
                z = fldmodel.ldmodel.get_input(batch, 'image')[0]
                rule_z = torch.zeros_like(z)
                rule_idx = fire.argmax(dim=-1)
                offset = 0
                for i in range(len(x)):
                    # train rule consequent
                    if rule_idx[i].item() !=  r:
                        continue
                    rule_z[offset] = z[i]
                    offset = offset + 1
                if offset == 0:
                    continue
                optimizer.zero_grad()
                z = z[:offset]
                z_fuzz = consequent_model(z)
                loss = criterion(z_fuzz, z)
                loss.backward()
                optimizer.step()
                
                batch_count = batch_count + 1
                if batch_count % step == 0:
                    x_p = fldmodel.ldmodel.decode_first_stage(z)
                    x_fuzz = fldmodel.ldmodel.decode_first_stage(z_fuzz)
                    # x_norm = img_norm(x)
                    x_p_norm = img_norm(x_p)
                    x_fuzz_norm = img_norm(x_fuzz)
                    combine_imgs2(x_p_norm,x_fuzz_norm)
                    csv_record(root_path+f"loss_preprocess_rule_{r+1}.csv", {'epoch': f"{e}",'batch': f"{batch_count}",'loss': f"{loss.item()}"})
        frozen_model(consequent_model)
    save_model(fldmodel.kbmodel, root_path+f"kbmodel_preprocess_rules{rule_num}.pt")
    logger.info("kbmodel preprocess done")
    
    # save boost train data of latent space
    dataset = LSUN('/home/yang/sda/github/fuzzydiffusion/output/datasets/lsun', 
                   dataset_sub_path=dataset_name,
                   phase='train', 
                   data_limit= sample_total_limit,
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), #https://blog.csdn.net/zylooooooooong/article/details/122805833
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    train_data = DataLoader(
        dataset, batch_size=batch_size, num_workers=5,shuffle=True, drop_last=True, pin_memory=True)
    count = 0
    c,h,w = z_shape
    boost_input = torch.empty([sample_batch_limit, batch_size, c, h, w]).to(options.device) # z_rec
    boost_fire = torch.empty([sample_batch_limit, batch_size, rule_num]).to(options.device) # fire
    boost_target = torch.empty([sample_batch_limit, batch_size, c, h, w]).to(options.device) # z
    for batch in tqdm(train_data):
        optimizer.zero_grad()
        x = batch['image'].to(options.device)
        x_fuzz, x_rec, z, z_rec, z_fuzz, fire = fldmodel(batch, latent_output=True)
        boost_input[count] = z_rec
        boost_fire[count] = fire
        boost_target[count] = z
        count = count + 1
    
    # kbmodel first train, single train stage , train rule consequent one by one
    logger.info("kbmodel first train, single train stage , train rule consequent one by one")
    optimizer = torch.optim.Adam(fldmodel.parameters(), lr=options.learning_rate, weight_decay=0)
    stat_model_param(fldmodel.kbmodel,"kbmodel")
    for r in range(rule_num):
        consequent_model = fldmodel.kbmodel[r]
        optimizer = torch.optim.Adam(consequent_model.parameters(), lr=options.learning_rate, weight_decay=0)
        activate_model(consequent_model)
        for e in tqdm(range(epoch), f"train rule_{r+1}"):
            batch_count = 0
            for batch in tqdm(train_data):
                x = batch['image'].to(options.device)
                fire = fldmodel.antecedent(x).to(options.device)
                z = fldmodel.ldmodel.get_input(batch, 'image')[0]
                rule_data = torch.zeros_like(z)
                rule_idx = fire.argmax(dim=-1)
                offset = 0
                for i in range(len(x)):
                    # train rule consequent
                    if rule_idx[i].item() !=  r:
                        continue
                    rule_data[offset] = z[i]
                    offset = offset + 1
                if offset == 0:
                    continue
                optimizer.zero_grad()
                z = rule_data[:offset]
                z_fuzz = consequent_model(z)
                loss = criterion(z_fuzz, z)
                loss.backward()
                optimizer.step()
                
                batch_count = batch_count + 1
                if batch_count % step == 0:
                    x_p = fldmodel.ldmodel.decode_first_stage(z)
                    x_fuzz = fldmodel.ldmodel.decode_first_stage(z_fuzz)
                    x_p_norm = img_norm(x_p)
                    x_fuzz_norm = img_norm(x_fuzz)
                    combine_imgs2(x_p_norm,x_fuzz_norm)
                    csv_record(root_path+f"loss_first_train_rule_{r+1}.csv", {'epoch': f"{e}",'batch': f"{batch_count}",'loss': f"{loss.item()}"})
        frozen_model(consequent_model)
    save_model(fldmodel.kbmodel, root_path+f"kbmodel_first_train_rules{rule_num}_epoch{epoch}.pt")
    logger.info("kbmodel first train done")
    
    # kbmodel second train , fusion train stage, train all consequents together
    logger.info("kbmodel second train , fusion train stage, train all consequents together")
    for i in range(rule_num):
        activate_model(fldmodel.kbmodel[i])
    stat_model_param(fldmodel, "fldmodel")
    optimizer = torch.optim.Adam(fldmodel.kbmodel.parameters(), lr=options.learning_rate, weight_decay=0)
    stat_model_param(fldmodel.kbmodel,"kbmodel")
    for _ in tqdm(range(epoch), f"train rules"):
        batch_count = 0
        for i in range(sample_batch_limit):
            fire = boost_fire[i]
            optimizer.zero_grad()
            z = boost_target[i]
            z_rec = boost_input[i]
            z_fuzz = fldmodel.knowledge_boost(z_rec, fire)
            loss = criterion(z_fuzz, boost_target[i])
            loss.backward()
            optimizer.step()
            
            batch_count = batch_count + 1
            if batch_count % step == 0:
                x_p = fldmodel.ldmodel.decode_first_stage(z)
                x_fuzz = fldmodel.ldmodel.decode_first_stage(z_fuzz)
                x_p_norm = img_norm(x_p)
                x_fuzz_norm = img_norm(x_fuzz)
                combine_imgs2(x_p_norm,x_fuzz_norm)
                csv_record(root_path+f"loss_second_train_rules{rule_num}.csv", {'epoch': f"{e}",'batch': f"{batch_count}",'loss': f"{loss.item()}"})
            
    save_model(fldmodel.kbmodel, root_path+f"kbmodel_second_train_rules{rule_num}_epoch{epoch}.pt")
    logger.info("kbmodel second train done")
    
    # kbmodel third train, inside of fldmodel train
    logger.info("kbmodel third train, inside of fldmodel train")
    optimizer = torch.optim.Adam(fldmodel.kbmodel.parameters(), lr=options.learning_rate, weight_decay=0)
    stat_model_param(fldmodel.kbmodel,"kbmodel")
    for _ in tqdm(range(epoch), f"train rules"):
        batch_count = 0
        for batch in tqdm(train_data):
            x = batch['image'].to(options.device)
            x_fuzz, x_rec, z, z_rec, z_fuzz, fire = fldmodel(batch, latent_output=True)
            optimizer.zero_grad()
            loss = criterion(z_fuzz, z)
            loss.backward()
            optimizer.step()
            
            batch_count = 0
            if batch_count % step ==0:
                x_p = fldmodel.ldmodel.decode_first_stage(z)
                x_fuzz = fldmodel.ldmodel.decode_first_stage(z_fuzz)
                x_norm = img_norm(x)
                x_p_norm = img_norm(x_p)
                x_rec_norm = img_norm(x_rec)
                x_fuzz_norm = img_norm(x_fuzz)
                combine_imgs4(x_norm, x_p_norm,x_rec_norm, x_fuzz_norm)
                csv_record(root_path+f"loss_third_train_rules{rule_num}.csv", {'epoch': f"{e}",'batch': f"{batch_count}",'loss': f"{loss.item()}"})
    save_model(fldmodel, root_path+f"kbmodel_third_train_rules{rule_num}_epoch{epoch}.pt")
    logger.info("kbmodel third train done")
    
    logger.info("lsun test start")
    lsun_test(dataset_name, fldmodel, root_path)
    logger.info("lsun test done")
    logger.remove(log_file)

def lsun_test(dataset_name, fldmodel, root_path):
    log_file = logger.add(root_path+'fuzzy-latent-diffusion-test-'+str(datetime.date.today()) +'.log')
    metrics_csv_path = root_path+'metrics-'+str(datetime.date.today()) +'.csv'
    dist_csv_path = root_path+'distribution-'+str(datetime.date.today()) +'.csv'
    
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
        x_fuzz, x_rec, z, z_rec, _, _ = fldmodel(batch,latent_output=True)
        
        x_p = fldmodel.ldmodel.decode_first_stage(z)
        
        # compress [-1,1]  into  [0,1]
        x_norm = sigmoid(x)
        x_p_norm = sigmoid(x_p)
        x_rec_norm =  sigmoid(x_rec)
        x_fuzz_norm =  sigmoid(x_fuzz)
        
        
        distributions = {}
        distributions['x'] = x
        distributions['z'] = z
        distributions['z_rec'] = z_rec
        distributions['x_rec'] = x_rec
        distributions['x_p'] = x_p
        distributions['x_fuzz'] = x_fuzz
        distributions['x_norm'] = x_norm
        distributions['x_p_norm'] = x_p_norm
        distributions['x_rec_norm'] = x_rec_norm
        distributions['x_fuzz_norm'] = x_fuzz_norm
        csv_dist_record(dist_csv_path, distributions)
        combine_imgs4(x_norm, x_p_norm,x_rec_norm,x_fuzz_norm)
        
        count = count + 1
        real_imgs = x_norm
        gen_imgs = x_fuzz_norm
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
    lsun_train('churches', "/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_churches-ldm-kl-8.yaml")