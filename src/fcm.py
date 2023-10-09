from fcmeans import FCM
import numpy as np
from loguru import logger
import os
import PIL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage.metrics import structural_similarity
from tqdm import tqdm
import torch

from setting import options
from  loader import LSUN

def img_clustering(imgs, n_clusters):
    fcm = FCM(n_clusters=n_clusters)
    b,c,h,w = np.array(imgs).shape
    img_data = np.array(imgs).reshape(b, -1)
    fcm.fit(img_data)
    centers = fcm.centers
    ssim = np.empty((b, n_clusters), dtype=float)
    for i in range(b):
        for j in range(n_clusters):
            im1 = (np.array(imgs[i])+ 1.0)*255/2
            im2 = (centers[j].reshape(c,h,w)+1.0)*255/2
            im1 = np.uint8(im1)
            im2 = np.uint8(im2)
            (score, diff) = structural_similarity(im1, im2, win_size=101, channel_axis=0, full=True)
            ssim[i][j] = score
    # print(ssim.max(axis=0))
    # print(np.argmax(ssim, axis=0))
    delegates_idx_list = np.argmax(ssim, axis=0).tolist()
    delegates_idx = set(delegates_idx_list) ## del repeating items
    n_dlg = len(delegates_idx)
    delegates = np.empty((n_dlg, c,h,w))
    for i in range(n_dlg):
        idx = delegates_idx.pop()
        delegates[i] = imgs[idx]
    return delegates

if __name__ == "__main__":
    dataset = LSUN('lsun', 'churches','train', transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]))
    train_data = DataLoader(
        dataset, batch_size=100, shuffle=True, drop_last=True, pin_memory=True)

    delegates = None
    for images,texts in tqdm(train_data):
        dlg = img_clustering(images, 3)
        if delegates is None:
            delegates = dlg
        else:
            delegates = np.concatenate((delegates, dlg), axis=0)
    local_delegates = torch.from_numpy(delegates)
    global_delegates = img_clustering(local_delegates, 3)
    count = 0
    ToImg = transforms.ToPILImage()
    for dlg in global_delegates:
        img = ToImg(dlg)
        count = count + 1
        img.save(options.base_path+"output/img/lsun/"+"lsun_churches_"+str(count)+".jpg") 