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

def read_coco():
    dataset_path = options.base_path+"output/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT/"
    dataset = read_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT", '')
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),"CoCo"):
        data = next(train_iter)
        #http://images.cocodataset.org/train2017/000000391895.jpg
        url = data['URL']
        text = data['TEXT']
        url_feilds = urlparse(url)
        filename = os.path.basename(url_feilds.path)
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            download(url,local_path, bar=bar_blank)  
        train_data[i] = [local_img, text]
    return train_data

def read_flowers():
    dataset_path = options.base_path+"output/datasets/nelorth/oxford-flowers/"
    fd = open(dataset_path+"/image_label.txt", "r")
    labels = []
    for line in fd.readlines():
        line.replace("\r", '')
        labels.append('a '+line + " flower")
    dataset = read_dataset("nelorth/oxford-flowers", '')
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),"flowers"):
        data = next(train_iter)
        img = data['image']
        text = labels[data['label']]
        filename = str(i)+".jpg"
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        train_data[i] = [local_img, text]

    test_len = dataset['test'].num_rows
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len),"flowers"):
        data = next(test_iter)
        img = data['image']
        text = data['label']
        filename = str(i)+".jpg"
        local_path = dataset_path+"test/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        test_data[i] = [local_img, text]
    
    part = int(test_len/2)
    valid_data = test_data[:part]
    test_data = test_data[part:]
    return train_data, valid_data, test_data

def read_cub200():
    dataset_path = options.base_path+"output/datasets/CUB2002011/"
    dataset_name = ['anjunhu/naively_captioned_CUB2002011_train', "anjunhu/naively_captioned_CUB2002011_test"]
    dataset = [read_dataset(dataset_name[0], ''), read_dataset(dataset_name[1], '')]
    train_len = dataset[0]['train'].num_rows + dataset[1]['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = chain(iter(dataset[0]['train']) , iter(dataset[1]['train']))
    for i in tqdm(range(train_len),"cub200"):
        data = next(train_iter)
        img = data['image']
        text = data['text']
        filename = str(i)+".jpg"
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        train_data[i] = [local_img, text]
    part = int(train_len /10)
    valid_data = train_data[train_len - 2* part : train_len - part]
    test_data = train_data[train_len - part : ]
    train_data = train_data[:train_len - 2* part]
    return train_data,valid_data, test_data

def read_pokemon():
    dataset_name = 'lambdalabs/pokemon-blip-captions'
    dataset_path = options.base_path+"output/datasets/"+dataset_name+'/'
    dataset = read_dataset(dataset_name, '')
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),"pokemon"):
        data = next(train_iter)
        img = data['image']
        text = data['text']
        filename = str(i)+".jpg"
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        train_data[i] = [local_img, text]
    part = int(train_len /10)
    valid_data = train_data[train_len - 2* part : train_len - part]
    test_data = train_data[train_len - part : ]
    train_data = train_data[:train_len - 2* part]
    return train_data,valid_data, test_data

def read_imagereward():
    dataset_name = 'THUDM/ImageRewardDB'
    dataset_path = options.base_path+"output/datasets/"+dataset_name+'/1k/'
    dataset = read_dataset(dataset_name, '1k')
    
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),"imagereward"):
        if i >= 6183:
            continue
        data = next(train_iter)
        img = data['image']
        text = data['prompt']
        filename = str(i)+".jpg"
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        train_data[i] = [local_img, text]
        
    valid_len = dataset['validation'].num_rows
    valid_data = np.empty([valid_len], dtype=int).tolist()
    valid_iter = iter(dataset['validation'])
    for i in tqdm(range(valid_len),"imagereward"):
        data = next(valid_iter)
        img = data['image']
        text = data['prompt']
        filename = str(i)+".jpg"
        local_path = dataset_path+"validation/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        valid_data[i] = [local_img, text]
    
    test_len = dataset['test'].num_rows
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len),"imagereward"):
        data = next(test_iter)
        img = data['image']
        text = data['prompt']
        filename = str(i)+".jpg"
        local_path = dataset_path+"test/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        test_data[i] = [local_img, text]
    return train_data,valid_data, test_data

def read_cifar10():
    dataset_name = 'cifar10'
    dataset_path = options.base_path+"output/datasets/"+dataset_name+'/'
    dataset = read_dataset(dataset_name, '')
    
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),"cifar10"):
        data = next(train_iter)
        img = data['img']
        text = data['label']
        filename = str(i)+".jpg"
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        train_data[i] = [local_img, text]
    
    test_len = dataset['test'].num_rows
    test_data = np.empty([test_len], dtype=int).tolist()
    test_iter = iter(dataset['test'])
    for i in tqdm(range(test_len),"cifar10"):
        data = next(test_iter)
        img = data['img']
        text = data['label']
        filename = str(i)+".jpg"
        local_path = dataset_path+"test/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        test_data[i] = [local_img, text]
    part = int(test_len/2)
    valid_data = test_data[:part]
    test_data = test_data[part:]
    return train_data, valid_data, test_data

def read_data(name):
    if name == 'coco':
        return read_coco()
    elif name == 'flowers':
        return read_flowers()
    elif name == 'pokemon':
        return read_pokemon()
    elif name == 'imagereward':
        return read_imagereward()
    elif name == 'cub200':
        return read_cub200()
    elif name == 'cifar10':
        return read_cifar10()
    else:
        logger.error("dataset %s not exist!" %(name))





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
    # read_coco()
    # read_flowers()
    # read_pokemon()
    # read_imagereward()
    # read_cub200()
    # read_cifar10()
    # dataset = CIFAR10("cifar10",transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    # CIFAR10("cifar10", phase="validation")
    # CIFAR10("cifar10", phase="test")
    # train_data = DataLoader(
    #     dataset, batch_size=options.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    dataset = COCO("ChristophSchuhmann/MS_COCO_2017_URL_TEXT",transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    # dataset = Flowers("nelorth/oxford-flowers",transform=transforms.Compose([
    #         transforms.Resize((32,32)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    
    # dataset = CUB200("anjunhu/naively_captioned_CUB2002011_train",transform=transforms.Compose([
    #         transforms.Resize((32,32)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    
    train_data = DataLoader(
        dataset, batch_size=options.batch_size, shuffle=True, drop_last=True, pin_memory=True)
    
    for image,text in train_data:
        image = image.to("cuda")
        image = image.to('cpu')
