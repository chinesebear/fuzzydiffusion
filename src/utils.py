import os
import random
import time

import torch
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PIL import Image
import csv

def stat_model_param(model, name=None):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if name is None:
        logger.info(f"total: {total_num:,}, trainable:{trainable_num:,}, frozen: {(total_num - trainable_num):,}")
    else:
        logger.info(f"[{name}] total: {total_num:,}, trainable:{trainable_num:,}, frozen: {(total_num - trainable_num):,}")
    return {'Total': total_num, 'Trainable': trainable_num}

def frozen_model(model):
    for param in model.parameters():
        param.requires_grad = False # Frozen 
    stat_model_param(model)
    
def activate_model(model):
    for param in model.parameters():
        param.requires_grad = True # trainable 
    stat_model_param(model)
    
def save_model(model, path):
    torch.save(model.state_dict(),path)
    logger.info(f"save {path} model parameters done")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"load {path} model parameters done")
        
def setup_seed(seed):
    # https://zhuanlan.zhihu.com/p/462570775
    torch.use_deterministic_algorithms(True) # 检查pytorch中有哪些不确定性
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 大于CUDA 10.2 需要设置
    logger.info("seed: %d, random:%.4f, torch random:%.4f, np random:%.4f" %(seed, random.random(), torch.rand(1), np.random.rand(1)))
    
def tensor_to_img(img_tensors):
    imgs = []
    toImg = transforms.ToPILImage()
    for tensor in img_tensors:
        img = toImg(tensor)
        imgs.append(img)
    return imgs

def img_norm(imgs):
    sigmoid = nn.Sigmoid()
    # compress [-1,1]  into  [0,1]
    imgs_norm =  sigmoid(imgs)
    return imgs_norm
 
def combine_imgs2(x1, x2):
    b,c,h,w = x1.shape
    
    x1_imgs = tensor_to_img(x1)
    x2_imgs = tensor_to_img(x2)
 
    # 创建空白长图
    result = Image.new("RGB", (w*2, h*b))
 
    # 拼接图片    
    for i in range(b):
        result.paste(x1_imgs[i], box=(0, i*h))
        result.paste(x2_imgs[i], box=(w, i*h))
 
    # 保存图片
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/test2.jpg")   

def combine_imgs3(x1, x2, x3):
    b,c,h,w = x1.shape
    
    x1_imgs = tensor_to_img(x1)
    x2_imgs = tensor_to_img(x2)
    x3_imgs = tensor_to_img(x3)
 
    # 创建空白长图
    result = Image.new("RGB", (w*3, h*b))
 
    # 拼接图片    
    for i in range(b):
        result.paste(x1_imgs[i], box=(0, i*h))
        result.paste(x2_imgs[i], box=(w, i*h))
        result.paste(x3_imgs[i], box=(w*2, i*h))
 
    # 保存图片
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/test3.jpg")

def combine_imgs4(x1, x2, x3, x4):
    b,c,h,w = x1.shape
    
    x1_imgs = tensor_to_img(x1)
    x2_imgs = tensor_to_img(x2)
    x3_imgs = tensor_to_img(x3)
    x4_imgs = tensor_to_img(x4)
 
    # 创建空白长图
    result = Image.new("RGB", (w*4, h*b))
 
    # 拼接图片    
    for i in range(b):
        result.paste(x1_imgs[i], box=(0, i*h))
        result.paste(x2_imgs[i], box=(w, i*h))
        result.paste(x3_imgs[i], box=(w*2, i*h))
        result.paste(x4_imgs[i], box=(w*3, i*h))
 
    # 保存图片
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/test4.jpg")

def combine_imgs5(x1, x2, x3, x4,x5):
    b,c,h,w = x1.shape
    
    x1_imgs = tensor_to_img(x1)
    x2_imgs = tensor_to_img(x2)
    x3_imgs = tensor_to_img(x3)
    x4_imgs = tensor_to_img(x4)
    x5_imgs = tensor_to_img(x5)
 
    # 创建空白长图
    result = Image.new("RGB", (w*5, h*b))
 
    # 拼接图片    
    for i in range(b):
        result.paste(x1_imgs[i], box=(0, i*h))
        result.paste(x2_imgs[i], box=(w, i*h))
        result.paste(x3_imgs[i], box=(w*2, i*h))
        result.paste(x4_imgs[i], box=(w*3, i*h))
        result.paste(x5_imgs[i], box=(w*4, i*h))
 
    # 保存图片
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/test5.jpg")

def combine_imgs5_with_prompt(x1, x2, x3, x4,x5, promprts):
    b,c,h,w = x1.shape
    
    x1_imgs = tensor_to_img(x1)
    x2_imgs = tensor_to_img(x2)
    x3_imgs = tensor_to_img(x3)
    x4_imgs = tensor_to_img(x4)
    x5_imgs = tensor_to_img(x5)
 
    # 创建空白长图
    result = Image.new("RGB", (w*5, h*b))
 
    # 拼接图片    
    for i in range(b):
        result.paste(x1_imgs[i], box=(0, i*h))
        result.paste(x2_imgs[i], box=(w, i*h))
        result.paste(x3_imgs[i], box=(w*2, i*h))
        result.paste(x4_imgs[i], box=(w*3, i*h))
        result.paste(x5_imgs[i], box=(w*4, i*h))
    
    unix = int(time.time())
    # 写提示词
    with open(f'/home/yang/sda/github/fuzzydiffusion/output/img/test/test5_{unix}.txt','w') as f:    #设置文件对象
        f.write('\n'.join(promprts))                 #将字符串写入文件中
    # 保存图片
    result.save(f"/home/yang/sda/github/fuzzydiffusion/output/img/test/test5_{unix}.jpg")
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/test5.jpg")

def combing_imgs(img_arr):
    col_num, row_num ,c,h,w = img_arr.shape
    result = Image.new('RGB', (w*col_num, h*row_num))
    for i in range(col_num):
        imgs = tensor_to_img(img_arr[i])
        for j in range(row_num):
             result.paste(imgs[i], box=(i*w, j*h))
    # 保存图片
    result.save("/home/yang/sda/github/fuzzydiffusion/output/img/combine.jpg")
                
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
    all_header = ['fid', 'is', 'mifid', 'kid',
              'psnr','ms_ssim','ssim',
              'precision', 'recall', 'epoch','batch','loss','clip']  
    row = []
    header = []
    for name in all_header:
        if name in data.keys():
            row.append(data[name])
            header.append(name)
            
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
    header = ['x', 'z','x_p','x_fuzz','x_norm', 
              "x_p_norm",'x_fuzz_norm']
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
            

class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, warm_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = warm_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
        
def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists, {path}")
        
def txt_split(name, path, split_num):
    txt_path = os.path.join(path, name)+".txt"  
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"File {txt_path} does not exist.")
    with open(txt_path, "r") as fd:
        lines = fd.readlines()
        fd.close()
    split_lines = len(lines) // split_num
    lines = [lines[i:i + split_lines] for i in range(0, len(lines), split_lines)]
    for i, line in enumerate(lines):
        with open(os.path.join(path, f"{name}_{i}.txt"), "w") as fd:
            fd.writelines(line)
            fd.close()
        logger.info(f"File {os.path.join(path, f'{name}_{i}.txt')} created.")
    return lines

def txt_combine(name, path, target_path):
    lines = []
    i = 0
    while True:
        txt_path = os.path.join(path, f"{name}_{i}.txt")
        if not os.path.exists(txt_path):
            break
        with open(txt_path, "r") as fd:
            lines += fd.readlines()
            fd.close()
        # os.remove(txt_path)
        # logger.info(f"File {txt_path} removed.")
        i += 1
    check_dir(target_path)
    with open(os.path.join(target_path, name)+".txt", "w") as fd:
        fd.writelines(lines)
        fd.close()
    logger.info(f"File {os.path.join(path, name)+'.txt'} created.")
    return lines
if __name__ == '__main__':
    print("done")