import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from fcmeans import FCM
import numpy as np
from loguru import logger
import PIL
from PIL import Image
from skimage.metrics import structural_similarity

from tqdm import tqdm
import csv
from collections import Counter

from setting import options
from  loader import LSUN, FFHQ,CELEBA_HQ,COCO
from metrics import Evaluator
from utils import check_dir

eval = Evaluator()
ToImg = transforms.ToPILImage()
sigmoid = nn.Sigmoid()

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def q_sample(x_start, t, noise, beta_1=1e-4, beta_T=0.02, T=1000):
    betas = torch.linspace(beta_1, beta_T, T).double()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    sqrt_alphas_bar =torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1. - alphas_bar)
    x_t = (extract(sqrt_alphas_bar, t, x_start.shape) * x_start +
            extract(sqrt_one_minus_alphas_bar, t, x_start.shape) * noise)
    return x_t

def calc_entropy(image):
    image = np.array(image)
    # 将图像转化为一维数组
    flat_image = image.flatten()
    # 计算每个像素值在数组中出现的频率
    freqs = dict(Counter(flat_image))
    # 计算每个像素值出现的概率
    probs = np.array(list(freqs.values())) / len(flat_image)
    # 计算熵
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def calc_entropys(imgs):
    b,c,h,w = imgs.shape
    entropys= np.empty([b])
    for i in range(b):
        img = ToImg(imgs[i])
        entropy = calc_entropy(img)
        entropys[i] = entropy
    return entropys

def calc_delta_entropy(img1, img2):
    entropy1 = calc_entropy(img1)
    entropy2 = calc_entropy(img2)
    dist = np.sqrt(np.sum((entropy1 - entropy2) ** 2))
    return dist

def diffusion_feature(imgs, n_step=4):
    b,c,h,w = imgs.shape
    timesteps = torch.arange(1,1000, step=1000//n_step).long()
    features = np.empty([b,n_step+1])
    for i in range(b):
        x_start = imgs[i]
        noise = torch.randn_like(x_start)
        entropys = np.empty(n_step+1)
        count = 1
        entropys[0] = calc_entropy(ToImg(sigmoid(x_start)))
        for step in timesteps:
            x_t = q_sample(x_start.view(1,c,h,w), step.view(1), noise.view(1,c,h,w))
            entropys[count] = calc_entropy(ToImg(sigmoid(x_t.view(c,h,w))))
            count = count + 1
        features[i] = entropys
    return features

def img_clustering(imgs, n_clusters):
    b,c,h,w = imgs.shape
    fcm = FCM(n_clusters=n_clusters)
    img_feature= diffusion_feature(imgs)
    fcm.fit(img_feature)
    centers = fcm.centers
    dist = np.empty((b, n_clusters), dtype=float)
    for i in range(b):
        for j in range(n_clusters):
            m = img_feature[i]
            n = centers[j]
            d = np.sqrt(np.sum(np.square(m - n)))
            dist[i][j] = d
    delegates_idx_list = np.argmin(dist, axis=0).tolist()
    delegates_idx = set(delegates_idx_list) ## del repeating items
    n_dlg = len(delegates_idx)
    delegates = torch.empty((n_dlg, c,h,w))
    for i in range(n_dlg):
        idx = delegates_idx.pop()
        delegates[i] = imgs[idx]
    return delegates

def get_delegates(train_data_loader, name, delegate_num_list):
    # train_data_loader shuffle is true, random
    local_delegates = None
    for i in tqdm(range(10)):
        batch = next(iter(train_data_loader))
        images = batch['image']
        dlg = img_clustering(images, 3)
        if local_delegates is None:
            local_delegates = dlg
        else:
            local_delegates = torch.concatenate((local_delegates, dlg), dim=0)
    for delegate_num in delegate_num_list:
        global_delegates = img_clustering(local_delegates, delegate_num)
        count = 0
        ToImg = transforms.ToPILImage()
        ds_path = options.base_path+f"output/delegates/{name}/"
        check_dir(ds_path)
        ds_csv_path = ds_path+f"{name}_{delegate_num}_delegates.csv"
        lsun_img_path_list = []
        for dlg in global_delegates:
            dlg = sigmoid(dlg*2) #[-1,1] -> [0,1]
            img = ToImg(dlg)
            count = count + 1
            lsun_img_path = ds_path+f"{name}_{delegate_num}_delegates_{count}.jpg"
            img.save(lsun_img_path) ## 经过transforms.Normalize归一化的图片
            lsun_img_path_list.append(lsun_img_path)
        with open(ds_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["delegates"])
            for path in lsun_img_path_list:
                writer.writerow([path])

def get_txtimg_delegates(train_data_loader, name, delegate_num_list):
    # train_data_loader shuffle is true, random
    local_delegates = None
    for i in tqdm(range(10)):
        batch = next(iter(train_data_loader))
        images = batch['image']
        dlg = img_clustering(images, 3)
        if local_delegates is None:
            local_delegates = dlg
        else:
            local_delegates = torch.concatenate((local_delegates, dlg), dim=0)
    for delegate_num in delegate_num_list:
        global_delegates = img_clustering(local_delegates, delegate_num)
        count = 0
        ToImg = transforms.ToPILImage()
        ds_path = options.base_path+f"output/delegates/{name}/"
        check_dir(ds_path)
        ds_csv_path = ds_path+f"{name}_{delegate_num}_delegates.csv"
        lsun_img_path_list = []
        for dlg in global_delegates:
            dlg = sigmoid(dlg*2) #[-1,1] -> [0,1]
            img = ToImg(dlg)
            count = count + 1
            lsun_img_path = ds_path+f"{name}_{delegate_num}_delegates_{count}.jpg"
            img.save(lsun_img_path) ## 经过transforms.Normalize归一化的图片
            lsun_img_path_list.append(lsun_img_path)
        with open(ds_csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["delegates"])
            for path in lsun_img_path_list:
                writer.writerow([path])


if __name__ == "__main__":
    # dataset = LSUN('lsun', 'churches','train', transform=transforms.Compose([
    #                 transforms.Resize((256,256)),
    #                 # transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #             ]),
    #             data_limit=None)
    # train_data_loader = DataLoader(
    #     dataset, batch_size=10, shuffle=True, drop_last=True, pin_memory=True)
    # get_delegates(train_data_loader,"lsun_church", [2,3,4,5,6,7,8,9,10])

    
    # dataset = LSUN('lsun', 'bedrooms','train', transform=transforms.Compose([
    #                 transforms.Resize((256,256)),
    #                 # transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #             ]),
    #             data_limit=None)
    # train_data_loader = DataLoader(
    #     dataset, batch_size=10, shuffle=True, drop_last=True, pin_memory=True)
    # get_delegates(train_data_loader,"lsun_bedroom", [2,3,4,5,6,7,8,9,10])
    
    dataset = COCO('ChristophSchuhmann/MS_COCO_2017_URL_TEXT', 
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    train_data_loader = DataLoader(
        dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    get_txtimg_delegates(train_data_loader,"coco", [2,3,4,5,6,7,8,9,10])
    
    print('done')