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




def bar_blank(current, total, width=80):
    return


def split_indices(length, phase, ratios=None):
    """Deterministic 7:1:2 split used in the paper (Table I)."""
    ratios = ratios if ratios is not None else options.split_ratio
    n_train = int(round(length * ratios[0]))
    n_val = int(round(length * ratios[1]))
    if phase == 'train':
        return list(range(0, n_train))
    if phase in ('validation', 'val'):
        return list(range(n_train, n_train + n_val))
    if phase in ('test', 'testing'):
        return list(range(n_train + n_val, length))
    raise ValueError(f"unknown phase: {phase}")


def read_dataset(name, subpath):
    if subpath == '':
        dataset_path = options.base_path+"datasets/"+name+"/"
    else:
        dataset_path = options.base_path+"datasets/"+name+"/"+subpath+"/"
    if os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    else :
        config = DownloadConfig(
            resume_download=True,
            max_retries=100,
            use_auth_token=os.environ.get("HF_TOKEN"),
        )
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
        # `ChristophSchuhmann/MS_COCO_2017_URL_TEXT` only has a `train` split
        # (591,753 images).  The paper splits the 592k images 7:1:2.
        split = self.dataset['train']
        indices = split_indices(split.num_rows, self.phase)
        self.data_len = len(indices)
        data_pairs = []
        for i in tqdm(indices, "data preload"):
            data = split[i]
            data_pairs.append([data['URL'], data['TEXT']])
        self.data_pairs = data_pairs
        self.local_dir = (
            options.base_path + "datasets/" + self.dataset_path + "/"
            + ("train" if self.phase == 'train' else self.phase) + "/"
        )
        os.makedirs(self.local_dir, exist_ok=True)
        logger.info("%s data preload done" %(self.dataset_path))

    def __getitem__(self, idx):
        url, text = self.data_pairs[idx]
        filename = os.path.basename(urlparse(url).path)
        img_path = self.local_dir + filename
        if not os.path.exists(img_path):
            download(url, self.local_dir, bar=bar_blank)
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

class CUB200(Dataset):
    """CUB-200-2011 (the paper's "CUB 200"/"CUE 200") from the local copy.

    The class names are used as the text condition ("a photo of a ...").
    Images are split 7:1:2 as in Table I.
    """

    def __init__(self, dataset_path=None, phase='train', transform=None,
                 data_limit=None):
        self.dataset_path = dataset_path or (
            options.base_path + "datasets/CUE-200-2011"
        )
        self.phase = phase
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.data_limit = data_limit
        self.data_pairs = []
        self.data_preload()

    @staticmethod
    def _read_index(path):
        entries = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.strip().split()
                if len(fields) >= 2:
                    entries[int(fields[0])] = fields[1]
        return entries

    def data_preload(self):
        logger.info("CUB200 data preload start")
        images = self._read_index(os.path.join(self.dataset_path, "images.txt"))
        labels = self._read_index(
            os.path.join(self.dataset_path, "image_class_labels.txt")
        )
        classes = self._read_index(os.path.join(self.dataset_path, "classes.txt"))
        class_names = {
            index: name.split(".", 1)[-1].replace("_", " ")
            for index, name in classes.items()
        }

        image_ids = sorted(images.keys())
        indices = split_indices(len(image_ids), self.phase)
        if self.data_limit is not None:
            indices = indices[:self.data_limit]

        image_root = os.path.join(self.dataset_path, "images")
        self.data_pairs = [
            (
                os.path.join(image_root, images[image_id]),
                "a photo of a " + class_names.get(int(labels[image_id]), "bird"),
            )
            for image_id in (image_ids[i] for i in indices)
        ]
        logger.info("CUB200 data preload done")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_path, text = self.data_pairs[idx]
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        img = image.copy()
        image.close()
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'text': text, 'caption': text}

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
        # The paper splits the full dataset 7:1:2 (Table I).  The released
        # folder contains all images under `<subset>_train/`, so every split
        # slices the same index list deterministically.
        img_txt_path = (
            options.base_path + "datasets/lsun/"
            + sub_map[self.dataset_sub_path] + '_train.txt'
        )
        if not os.path.exists(img_txt_path):
            logger.error(f"{img_txt_path} not exists")
            return
        with open(img_txt_path, "r") as fd:
            lines = [line.strip() for line in fd if line.strip()]

        indices = split_indices(len(lines), self.phase)
        if self.data_limit is not None:
            indices = indices[:self.data_limit]
        self.data_len = len(indices)

        image_dir = (
            options.base_path + "datasets/lsun/"
            + self.dataset_sub_path + "_train/"
        )
        data_pairs = []
        for i in tqdm(indices):
            img_path = image_dir + lines[i]
            if len(data_pairs) < 100 and not os.path.exists(img_path):
                logger.error(f"{img_path} not exists")
                return
            data_pairs.append([img_path, ''])
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
