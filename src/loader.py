import os
from abc import ABC, abstractmethod

import numpy as np
import torch
# from torchtext.data import get_tokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# from torchtext.data import get_tokenizer
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt
from loguru import logger
from setting import options
from datasets import load_dataset,load_from_disk,DownloadConfig
from tqdm import tqdm
import pandas as pd
import random
import jsonlines
from translate.storage.tmx import tmxfile
from urllib.parse import urlparse
from wget import download
from PIL import Image
from scipy.io import loadmat
from itertools import chain


from setting import options



def bar_blank(current, total, width=80):
    return

def read_dataset(name, subpath):
    if subpath == '':
        dataset_path = options.base_path+"datasets/"+name+"/"
    else:
        dataset_path = options.base_path+"datasets/"+name+"/"+subpath+"/"
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
        self.dataset = read_dataset(self.dataset_path, '')
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
            local_path = options.base_path+"datasets/"+self.dataset_path+"/train/"
            local_img = local_path+filename
            if not os.path.exists(local_img):
                download(url,local_path, bar=bar_blank)
            data_pairs[i] = [local_img, text]
        self.data_pairs = data_pairs
        logger.info("%s data preload done" %(self.dataset_path))
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path,text = self.data_pairs[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        img = image.copy()
        image.close()
        if self.transform:
            img = self.transform(img)
        item = {'image': img, 'text': text, 'caption': text} # img:torch.tensor c,h,w
        return item

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

def tensor_to_img(data):
    data = np.uint8((data+1.0)*255/2)
    c,h,w = data.shape
    data_rgb = data.reshape(h,w,c)
    img = Image.fromarray(data_rgb)
    return img

class FFHQ(Dataset):
    def __init__(self, dataset_path, dataset_sub_path='', phase='train', transform=None, target_transform=None, data_limit=None):
        self.dataset_path = dataset_path
        self.dataset_sub_path = dataset_sub_path # churches,bedrooms,cats
        self.dataset = None
        self.phase = phase # train validation test
        self.data_len = 0
        self.data_limit = data_limit
        self.data_pairs = []
        self.transform = transform
        self.target_transform = target_transform
        self.data_preload()
    
    def data_preload(self):
        logger.info(f"FFHQ data preload start")
        imgs_path = self.dataset_path
        ld = os.listdir(imgs_path)
        img_total = 0
        for file in ld:
            if file.endswith(".png"):
                img_total = img_total + 1
        if self.data_limit is None:
            self.data_len = img_total
        elif img_total > self.data_limit:
            self.data_len = self.data_limit
        else:
            self.data_len = img_total
        logger.info(f"FFHQ data size:{self.data_len}")
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        idx = 0
        for file in tqdm(ld): 
            if file.endswith(".png"):
                img_path = imgs_path+f"/{file}"
                if idx < 1000 and not os.path.exists(img_path):
                    logger.error(f"{img_path} not exists")
                    return
                if idx >= self.data_len:
                    break
                text = ''
                data_pairs[idx] = [img_path, text]
                idx = idx + 1
        self.data_pairs = data_pairs
        logger.info(f"FFHQ data preload done")
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path,text = self.data_pairs[idx]
        image = Image.open(img_path)
        img = image.copy()
        image.close()
        if self.transform:
            img = self.transform(img)
        item = {'image': img, 'text': ''} # img:torch.tensor c,h,w
        return item

class CELEBA_HQ(Dataset):
    def __init__(self, dataset_path, dataset_sub_path='', phase='train', transform=None, target_transform=None, data_limit=None):
        self.dataset_path = dataset_path
        self.dataset_sub_path = dataset_sub_path # churches,bedrooms,cats
        self.dataset = None
        self.phase = phase # train validation test
        self.data_len = 0
        self.data_limit = data_limit
        self.data_pairs = []
        self.transform = transform
        self.target_transform = target_transform
        self.data_preload()
    
    def data_preload(self):
        logger.info(f"CELEBA_HQ data preload start")
        imgs_path = self.dataset_path
        ld = os.listdir(imgs_path)
        img_total = 0
        for file in ld:
            if file.endswith(".jpg"):
                img_total = img_total + 1
        if self.data_limit is None:
            self.data_len = img_total
        elif img_total > self.data_limit:
            self.data_len = self.data_limit
        else:
            self.data_len = img_total
        logger.info(f"CELEBA_HQ data size:{self.data_len}")
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        idx = 0
        for file in tqdm(ld): 
            if file.endswith(".jpg"):
                img_path = imgs_path+f"/{file}"
                if idx < 1000 and not os.path.exists(img_path):
                    logger.error(f"{img_path} not exists")
                    return
                if idx >= self.data_len:
                    break
                text = ''
                data_pairs[idx] = [img_path, text]
                idx = idx + 1
        self.data_pairs = data_pairs
        logger.info(f"CELEBA_HQ data preload done")
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path,text = self.data_pairs[idx]
        image = Image.open(img_path)
        img = image.copy()
        image.close()
        if self.transform:
            img = self.transform(img)
        item = {'image': img, 'text': ''} # img:torch.tensor c,h,w
        return item

class LSUN(Dataset):
    def __init__(self, dataset_path, dataset_sub_path='', phase='train', transform=None, target_transform=None, data_limit=None):
        self.dataset_path = dataset_path
        self.dataset_sub_path = dataset_sub_path # churches,bedrooms,cats
        self.dataset = None
        self.phase = phase # train validation test
        self.data_len = 0
        self.data_limit = data_limit
        self.data_pairs = []
        self.transform = transform
        self.target_transform = target_transform
        self.data_preload()
    
    def data_preload(self):
        logger.info(f"LSUN/{self.dataset_sub_path}-{self.phase} data preload start")
        sub_map = {"churches":"church_outdoor", "cats":"cat", "bedrooms":"bedrooms"}
        img_txt_name = sub_map[self.dataset_sub_path]+'_'+self.phase+".txt"
        img_txt_path = options.base_path+"datasets/lsun/"+img_txt_name
        if not os.path.exists(img_txt_path):
            logger.error(f"{img_txt_path} not exists")
            return
        fd = open(img_txt_path, "r")
        lines = fd.readlines()
        fd.close()
        if self.data_limit is None:
            self.data_len = len(lines)
        elif len(lines) > self.data_limit:
            self.data_len = self.data_limit
        else:
            self.data_len = len(lines)
        data_pairs = np.empty([self.data_len], dtype=int).tolist()
        for i in tqdm(range(self.data_len)):
            img_path = options.base_path+f"datasets/lsun/{self.dataset_sub_path}_{self.phase}/"+ lines[i].replace('\n', '')
            if i < 100 and not os.path.exists(img_path):
                logger.error(f"{img_path} not exists")
                return
            text = ''
            data_pairs[i] = [img_path, text]
        self.data_pairs = data_pairs
        logger.info(f"LSUN/{self.dataset_sub_path}-{self.phase} data preload done")
    
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        img_path,text = self.data_pairs[idx]
        image = Image.open(img_path)
        img = image.copy()
        image.close()
        if self.transform:
            img = self.transform(img)
            img = img
        item = {'image': img, 'text': ''} # img:torch.tensor c,h,w
        return item

class Vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<sos>":options.SOS, "<eos>":options.EOS, "<pad>":options.PAD,"<unk>":options.UNK}
        self.word2count = {"<sos>":1, "<eos>":1, "<pad>":1,"<unk>":1}
        self.index2word = {options.SOS: "<sos>", options.EOS: "<eos>", options.PAD:"<pad>",options.UNK: "<unk>"}
        self.n_words = 4  # Count PAD , SOS and EOS
        self.feature_max = [] # max value of feature
        self.feature_min = [] # min value of feature
        self.line_max = 0 # max length of sentence

    def addTokens(self, tokens):
        for word in tokens:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
if __name__ == '__main__':
    # dataset = CIFAR10("cifar10",transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    # CIFAR10("cifar10", phase="validation")
    # CIFAR10("cifar10", phase="test")
    # train_data = DataLoader(
    #     dataset, batch_size=options.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    
    
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
    

            
    # train_data = DataLoader(
    #     dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    # ToImg = transforms.ToPILImage()
    # for data in train_data:
    #     imgs = data['image']
    #     b,c,h,w = imgs.shape
    #     img = ToImg(imgs[0])
    #     img.save("/home/yang/sda/github/fuzzydiffusion/output/img/loader.jpg")
    
    # dataset = FFHQ('/home/yang/sda/github/fuzzydiffusion/datasets/FFHQ/', 
    #                transform=transforms.Compose([
    #                 transforms.Resize((256,256)),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #             ]))
    # train_data = DataLoader(
    #     dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    # for data in train_data:
    #     imgs = data['image']
    #     print(imgs.shape)
    
    # dataset = CELEBA_HQ('/home/yang/sda/github/fuzzydiffusion/datasets/celeba_hq_256',
    #                     # data_limit=5000, 
    #                     transform=transforms.Compose([
    #                         transforms.Resize((256,256)),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                     ]))
    # train_data = DataLoader(
    #     dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    # for data in train_data:
    #     imgs = data['image']
    #     print(imgs.shape)
    
    # dataset = CELEBA_HQ('/home/yang/sda/github/fuzzydiffusion/datasets/celeba_hq_256',
    #                     # data_limit=5000, 
    #                     transform=transforms.Compose([
    #                         transforms.Resize((256,256)),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #                     ]))
    # train_data = DataLoader(
    #     dataset, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    # for data in train_data:
    #     imgs = data['image']
    #     print(imgs.shape)
    dataset = COCO("ChristophSchuhmann/MS_COCO_2017_URL_TEXT",transform=transforms.Compose([
            transforms.Resize((32,32)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    
    dataset = LSUN(options.base_path+'datasets/lsun', 
                   dataset_sub_path='bedrooms',
                   phase='train', 
                   data_limit=None,
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
    dataset = LSUN(options.base_path+'datasets/lsun', 
                   dataset_sub_path='churches',
                   phase='train', 
                   data_limit=None,
                   transform=transforms.Compose([
                    transforms.Resize((256,256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]))
   
    print("done")
