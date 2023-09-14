import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torchtext.data import get_tokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from loguru import logger
from setting import options
from datasets import load_dataset,load_from_disk,DownloadConfig
from tqdm import tqdm
import random
import jsonlines
from translate.storage.tmx import tmxfile
from urllib.parse import urlparse
from wget import download
from PIL import Image
from scipy.io import loadmat
from itertools import chain





def bar_blank(current, total, width=80):
    return

def read_dataset(name, subpath):
    dataset_path = options.base_path+"output/datasets/"+name+"/"+subpath+"/"
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else :
        config = DownloadConfig(resume_download=True, max_retries=100,use_auth_token='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') 
        if subpath == '':
            dataset = load_dataset(name, download_config=config)
        else:
            dataset = load_dataset(name,subpath, download_config=config)
        dataset.save_to_disk(dataset_path)
    logger.info("%s-%s done" %(name,subpath))
    return dataset

class BaseDataSet(Dataset,ABC):
    def __init__(self, dataset_path, dataset_sub_path='', phase='train', transform=None, target_transform=None):
        self.dataset_path = dataset_path
        self.dataset = read_dataset(dataset_path, dataset_sub_path)
        self.phase = phase # train validation test
        self.data_len = 0
        self.data_pairs = []
        self.transform = transform
        self.target_transform = target_transform
        self.data_preload()
    
    @abstractmethod
    def data_preload(self):
        """
        你需要在子类中实现该方法, 子类才允许被实例化
        """

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img,text = self.data_pairs[idx]
        if self.transform:
            img = self.transform(img)
        return img, text

class CIFAR10(BaseDataSet):
    def data_preload(self):
        logger.info("%s data preload start" %(self.dataset_path))
        if not self.phase in self.dataset:
            logger.error("%s data not exist" %(self.phase))
            return
        self.data_len = self.dataset[self.phase].num_rows
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        data_iter = iter(self.dataset[self.phase])
        for i in tqdm(range(self.data_len), "data preload"):
            data = next(data_iter)
            img = data['img']
            text = data['label']
            data_pairs[i] = [img, text]
        self.data_pairs = data_pairs
        logger.info("%s data preload done" %(self.dataset_path))

class COCO(BaseDataSet):
    def data_preload(self):
        logger.info("%s data preload start" %(self.dataset_path))
        if not self.phase in self.dataset:
            logger.error("%s data not exist" %(self.phase))
            return
        self.data_len = self.dataset[self.phase].num_rows
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        data_iter = iter(self.dataset[self.phase])
        for i in tqdm(range(self.data_len), "data preload"):
            data = next(data_iter)
            url = data['URL']
            text = data['TEXT']
            url_feilds = urlparse(url)
            filename = os.path.basename(url_feilds.path)
            local_path = options.base_path+"output/datasets/"+self.dataset_path+"/train/"
            local_img = local_path+filename
            if not os.path.exists(local_img):
                download(url,local_path, bar=bar_blank)
            img = Image.open(local_img)
            if img.mode != 'RGB':
                img = img.convert("RGB")
            data_pairs[i] = [img, text]
        self.data_pairs = data_pairs
        logger.info("%s data preload done" %(self.dataset_path))

class Flowers(BaseDataSet):
    def data_preload(self):
        logger.info("%s data preload start" %(self.dataset_path))
        if not self.phase in self.dataset:
            logger.error("%s data not exist" %(self.phase))
            return
        label_path = options.base_path+"doc/"
        fd = open(label_path+"image_label.txt", "r")
        labels = []
        for line in fd.readlines():
            line = line.replace('\r', '')
            line = line.replace('\n', '')
            line = line.replace('\'', '')
            labels.append('a'+line + " flower")
        fd.close()
        self.data_len = self.dataset[self.phase].num_rows
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        data_iter = iter(self.dataset[self.phase])
        for i in tqdm(range(self.data_len), "data preload"):
            data = next(data_iter)
            img = data['image']
            text = labels[data['label']]
            data_pairs[i] = [img, text]
        self.data_pairs = data_pairs
        logger.info("%s data preload done" %(self.dataset_path))

class CUB200(BaseDataSet):
    def data_preload(self):
        logger.info("CUB200 data preload start")
        if not self.phase in self.dataset:
            logger.error("%s data not exist" %(self.phase))
            return
        dataset_path = "anjunhu/naively_captioned_CUB2002011_test"
        dataset = [self.dataset, read_dataset(dataset_path, '')]
        self.data_len = dataset[0]['train'].num_rows + dataset[1]['train'].num_rows
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        data_iter = chain(iter(dataset[0]['train']) , iter(dataset[1]['train']))
        for i in tqdm(range(self.data_len),"data preload"):
            data = next(data_iter)
            img = data['image']
            text = data['text']
            data_pairs[i] = [img, text]
        self.data_pairs = data_pairs
        logger.info("CUB200 data preload done")

if __name__ == '__main__':
    dataset = CIFAR10("cifar10",transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    CIFAR10("cifar10", phase="validation")
    CIFAR10("cifar10", phase="test")
    train_data = DataLoader(
        dataset, batch_size=options.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    dataset = COCO("ChristophSchuhmann/MS_COCO_2017_URL_TEXT",transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    dataset = Flowers("nelorth/oxford-flowers",transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    dataset = CUB200("anjunhu/naively_captioned_CUB2002011_train",transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    train_data = DataLoader(
        dataset, batch_size=options.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    
    for image,text in train_data:
        image = image.to("cuda")
        image = image.to('cpu')
