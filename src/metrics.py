import torch
from torch import nn
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.functional.multimodal import clip_score
from torchvision.transforms import ToTensor,ToPILImage,Resize
from torchvision import transforms

import numpy as np
from functools import partial
import evaluate
from PIL import Image
import csv

from setting import options


#https://github.com/NVlabs/stylegan2-ada-pytorch/tree/main/metrics
def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None


def knn_precision_recall_features( ref_features, eval_features , nhood_size=3, row_batch_size=10000, col_batch_size=50000,num_gpus=1, rank=0):
    real_features = ref_features
    gen_features = eval_features

    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if rank == 0 else None)
        kth = torch.cat(kth) if rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=num_gpus, rank=rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if rank == 0 else 'nan')
    return results['precision'], results['recall']

class Evaluator(nn.Module):
    def __init__(self, device=options.device):
        super().__init__()
        self.device = device
        self.fid_metrics = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.mifid_metrics = MemorizationInformedFrechetInceptionDistance(feature=2048, normalize=True).to(device)
        self.clip_metrics = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        self.inception_metrics = InceptionScore(normalize=True).to(device)
        self.kid_metrics = KernelInceptionDistance(subset_size=3, normalize=True).to(device)
        self.psnr_metrics = PeakSignalNoiseRatio().to(options.device)
        self.ms_ssim_metrics = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.ssim_metrics = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        self.vgg16.eval()
        self.vgg16.to(device)
        self.to_tensor = ToTensor()
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.to_image = transforms.ToPILImage()
        
    def resize299(self, imgs):
        b,c,h,w = imgs.shape
        imgs_out = torch.empty((b,c,299,299)).to(self.device)
        for i in range(b):
            imgs_out[i] = self.preprocess(imgs[i])
        return imgs_out
        

    def calc_fid(self, real_imgs, fake_imgs):
        # real_imgs = self.resize299(real_imgs)
        # fake_imgs = self.resize299(fake_imgs)
        self.fid_metrics.update(real_imgs, real=True)
        self.fid_metrics.update(fake_imgs, real=False)
        fid_score = float(self.fid_metrics.compute())
        # self.fid_metrics.reset()
        return round(fid_score,2)
    
    def calc_mifid(self, real_imgs, fake_imgs):
        real_imgs = self.resize299(real_imgs)
        fake_imgs = self.resize299(fake_imgs)
        self.mifid_metrics.update(real_imgs, real=True)
        self.mifid_metrics.update(fake_imgs, real=False)
        fid_score = float(self.mifid_metrics.compute())
        return round(fid_score,2)
    
    def calc_is(self, gen_imgs):
        gen_imgs = self.resize299(gen_imgs)
        self.inception_metrics.update(gen_imgs)
        is_mean, is_std = self.inception_metrics.compute()
        is_score = is_mean
        return round(is_score.item(),2)
    
    def calc_kid(self, real_imgs, fake_imgs):
        real_imgs = self.resize299(real_imgs)
        fake_imgs = self.resize299(fake_imgs)
        self.kid_metrics.update(real_imgs, real=True)
        self.kid_metrics.update(fake_imgs, real=False)
        kid_mean,kid_std = self.kid_metrics.compute()
        return round(kid_mean.item(),2)
        
    def calc_psnr(self, real_imgs, fake_imgs):
        psnr = self.psnr_metrics(real_imgs, fake_imgs)
        return round(psnr.item(),2)
    
    def calc_ms_ssim(self, real_imgs, fake_imgs):
        ms_ssim = self.ms_ssim_metrics(real_imgs, fake_imgs)
        return round(ms_ssim.item(),2)
    
    def calc_ssim(self, real_imgs, fake_imgs):
        ssim = self.ssim_metrics(real_imgs, fake_imgs)
        return round(ssim.item(),2)
    
    def calc_ssim2(self, real_img, fake_img):
        c,h,w = real_img.shape
        real_imgs = real_img.view(1,c,h,w)
        fake_imgs = fake_img.view(1,c,h,w)
        ssim = self.ssim_metrics(real_imgs, fake_imgs)
        return ssim
    
    def calc_clip(self, images, prompts):
        # b = len(images)
        # c,h,w = self.to_tensor(images[0]).shape
        # imgs = torch.empty((b, c,h,w))
        # for i in range(b):
        #     img = np.array(images[i]).reshape(c,h,w)
        #     img_tensor = torch.from_numpy(img).cuda()
        #     imgs[i] = img_tensor
        clip_score = self.clip_metrics(images, prompts).detach() # needs time
        clip_score = round(float(clip_score), 4)
        return round(clip_score,2)
    
    def calc_preision_recall(self, ref_imgs, eval_imgs):
        with torch.no_grad():
            ref_features = self.vgg16(ref_imgs)
            eval_features = self.vgg16(eval_imgs)
        precision, recall = knn_precision_recall_features(ref_features, eval_features)
        return round(precision,2),round(recall,2)
    
if __name__ == '__main__':
    eval = Evaluator()
    img1 = Image.open("/home/yang/sda/github/fuzzydiffusion/doc/test/unconditional/inputs_0.jpg")
    img2 = Image.open("/home/yang/sda/github/fuzzydiffusion/doc/test/unconditional/reconstruction_0.jpg")
    
    # acc = eval.calc_accuracy(img1, img2)
    # print(acc)
    # prec = eval.calc_precision(img1, img2)
    # print(prec)
    # recall = eval.calc_recall(img1, img2)
    # print(recall)
    
    test_path="/home/yang/sda/github/fuzzydiffusion/doc/test/unconditional/test.csv"
    real_imgs = []
    fake_imgs = []
    with open(test_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # print(headers)
        for row in reader:
            img_path = row[0]
            real_img = Image.open(img_path)
            real_imgs.append(real_img)
            img_path = row[1]
            fake_img = Image.open(img_path)
            fake_imgs.append(fake_img)
    fid = eval.calc_fid(real_imgs, fake_imgs)
    print(f"fid:{fid}")
    
    precision,recall = eval.calc_preision_recall(real_imgs, fake_imgs)
    print(f"precision:{precision}")
    print(f"recall:{recall}")
    
    test_path="/home/yang/sda/github/fuzzydiffusion/doc/test/text-conditional/test.csv"
    imgs=[]
    prompts=[]
    resize = Resize((256,256))
    with open(test_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        # print(headers)
        for row in reader:
            img_path = row[0]
            img1 = Image.open(img_path)
            img_path = row[1]
            img2 = Image.open(img_path)
            imgs.append(resize(img1))
            text = row[2]
            prompts.append(text)
    clip = eval.calc_clip(imgs, prompts)
    print(f"clip:{clip}")