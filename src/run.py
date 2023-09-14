import os
import datetime
import time

import torch
import torch.optim as optim
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from loguru import logger
from PIL import Image
import numpy as np

from model import GaussianDiffusionSampler, GaussianDiffusionTrainer, UNet
from scheduler import GradualWarmupScheduler
from setting import options,setting_info
from loader import CIFAR10,Flowers

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

start_time = time.time()

def func_start():
    global start_time
    start_time = time.time()

def func_cost():
    global start_time
    cur_time = time.time()
    delta_time = cur_time - start_time
    start_time = time.time()
    return int(delta_time)

def save_model(model,file):
    path = options.model_parameter_path+file+"-"+str(datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"))+".pth"
    torch.save(model.state_dict(), path)
    path = options.model_parameter_path+file+".pth"
    torch.save(model.state_dict(), path)
    logger.info("save %s model parameters done, %s" %(file, path))

def load_model(model, file):
    path = options.model_parameter_path+file+".pth"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logger.info("load %s model parameters done, %s." %(file, path))
    else:
        logger.warning("load %s model parameters fail, %s." %(file, path))
    
def save_img(img, file):
    path = options.img_path+file+".jpg"
    img = Image.fromarray(img)
    img.save(path)
    logger.info("save %s image done, %s." %(file, path))

def img_to_tensor(images):
    imgs = []
    for img in images:
        img = np.array(img)
        H, W, C = img.shape
        img = img.reshape((C, H, W))
        imgs.append(img)
    imgs = np.array(imgs)
    x_0 = torch.from_numpy(imgs).to(options.device).float()
    x_0 = x_0 / 127.5 - 1.0  #[-1.0, 1.0]
    return x_0

def train():
    log_file = logger.add(options.base_path+"output/log/train-"+str(datetime.date.today()) +'.log')
    # dataset
    dataset = Flowers("nelorth/oxford-flowers",transform=transforms.Compose([
            transforms.RandomCrop((500,500)),
            transforms.Resize(options.img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    train_data = DataLoader(
        dataset, batch_size=options.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    # model setup
    net_model = UNet(T=options.T, ch=options.unet.channel, ch_mult=options.unet.channel_mult, attn=options.unet.attn,
                     num_res_blocks=options.unet.num_res_blocks, dropout=options.unet.dropout).to(options.device)
    logger.info("[model setting] %s" %(setting_info()))
    # load_model(net_model, "diffusion")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=options.learning_rate, weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=options.epoch, eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=options.multiplier, warm_epoch=options.epoch // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, options.diff.beta_1, options.diff.beta_T, options.T).to(options.device)
    # start training
    for i in range(options.epoch):
        count = 0
        step = int(len(train_data) / 5)
        total_loss = 0
        func_start()
        for images, text in train_data :
            # train
            optimizer.zero_grad()
            x_0 = images.to(options.device)
            loss = trainer(x_0).sum() / 1000.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), options.grad_clip)
            optimizer.step()
            total_loss = total_loss + loss.item()
            count = count + 1
            if count % step ==0:
                logger.info("epoch %d , train loss: %.4f , progress: %d%%" %(i, total_loss/count, int(count * 100 /len(train_data))))
        warmUpScheduler.step()
        epoch_cost = func_cost()
        remaining_time = epoch_cost*(options.epoch - i - 1)
        logger.info("epoch time: %s seconds, remaining time: %s" %(str(epoch_cost), str(datetime.timedelta(seconds=remaining_time))))
    save_model(net_model, "diffusion")
    logger.remove(log_file)

def eval():
    log_file = logger.add(options.base_path+"output/log/eval-"+str(datetime.date.today()) +'.log')
    with torch.no_grad(): # aganist GPU memory up
        # load model and evaluate
        model = UNet(T=options.T, ch=options.unet.channel, ch_mult=options.unet.channel_mult, attn=options.unet.attn,
                        num_res_blocks=options.unet.num_res_blocks, dropout=0.1)
        logger.info("[model setting] %s" %(setting_info()))
        load_model(model, "diffusion")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, options.diff.beta_1, options.diff.beta_T, options.T).to(options.device)
        epoch = 8
        for i in tqdm(range(epoch),'sampling'):
            # Sampled from standard normal distribution
            noisyImage = torch.randn(
                size=[options.batch_size, 3, options.img_width, options.img_height], device=options.device)
            saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
            save_image(saveNoisy, options.img_path+str(i)+"_noisy.jpg")
            sampledImgs = sampler(noisyImage)
            sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
            save_image(sampledImgs, options.img_path+str(i)+"_sampled.jpg")
    logger.remove(log_file)
        
if __name__ == '__main__':
    train()
    eval()