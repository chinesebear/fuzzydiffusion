import os
import datetime

import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from loguru import logger
from PIL import Image
import numpy as np

from model import GaussianDiffusionSampler, GaussianDiffusionTrainer, UNet
from scheduler import GradualWarmupScheduler
from setting import options
from loader import read_flowers,read_cifar10


def save_model(model,file):
    path = options.model_parameter_path+file+".pth"
    torch.save(model.state_dict(), path)
    logger.info("save %s model parameters done, %s" %(file, path))

def load_model(model, file):
    path = options.model_parameter_path+file+".pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info("load %s model parameters done, %s." %(file, path))
    else:
        logger.error("load %s model parameters fail, %s." %(file, path))
    
def save_img(img, file):
    path = options.img_path+file+".png"
    img = Image.fromarray(img)
    img.save(path)
    logger.info("save %s image done, %s." %(file, path))

def train():
    log_file = logger.add(options.base_path+"output/log/train-"+str(datetime.date.today()) +'.log')
    # dataset
    train_data,_,_ = read_flowers()

    # model setup
    net_model = UNet(T=options.T, ch=options.unet.channel, ch_mult=options.unet.channel_mult, attn=options.unet.attn,
                     num_res_blocks=options.unet.num_res_blocks, dropout=options.unet.dropout).to(options.device)
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=options.learning_rate, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=200, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=2.0, warm_epoch=20, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, options.diff.beta_1, options.diff.beta_T, options.T).to(options.device)

    # start training
    for i in range(options.epoch):
        count = 0
        step = int(len(train_data) / 100)
        total_loss = 0
        for image, label in train_data:
            # train
            optimizer.zero_grad()
            img = Image.open(image).resize((32,32))
            img = np.array(img) 
            H, W, C = img.shape
            x_0 = torch.from_numpy(img).view(1, C, H, W).to(options.device).float()
            loss = trainer(x_0).sum() / 1000.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), 1.0)
            optimizer.step()
            total_loss = total_loss + loss.item()
            count = count + 1
            if count % step ==0:
                logger.info("epoch %d , train loss: %.4f , progress: %d%%" %(i, total_loss/count, int(count * 100 /len(train_data))))
        warmUpScheduler.step()
    save_model(net_model, "fuzzydiffusion")
    logger.remove(log_file)

def eval():
    # load model and evaluate
    with torch.no_grad():
        model = UNet(T=options.T, ch=options.unet.channel, ch_mult=options.unet.channel_mult, attn=options.unet.attn,
                     num_res_blocks=options.unet.num_res_blocks, dropout=0.)
        load_model(model, "fuzzydiffusion")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, options.diff.beta_1, options.diff.beta_T, options.T).to(options.device)
        # Sampled from standard normal distribution
        noisyImage = torch.randn(
            size=[options.batch_size, 3, 32, 32], device=options.device)
        saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        save_image(saveNoisy, "noisy")
        sampledImgs = sampler(noisyImage)
        sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        save_image(sampledImgs, "sampled")
        
if __name__ == '__main__':
    train()