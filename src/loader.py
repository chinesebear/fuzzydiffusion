import os
import numpy as np
from torchtext.data import get_tokenizer
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
    dataset = read_dataset("nelorth/oxford-flowers", '')
    train_len = dataset['train'].num_rows
    train_data = np.empty([train_len], dtype=int).tolist()
    train_iter = iter(dataset['train'])
    for i in tqdm(range(train_len),"Flowers"):
        data = next(train_iter)
        img = data['image']
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
    for i in tqdm(range(test_len),"Flowers"):
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
    train_iter = iter(dataset[0]['train'] + dataset[1]['train'])
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
    return train_data

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
    return train_data

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
        local_path = dataset_path+"train/"
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
        local_path = dataset_path+"train/"
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
        img = data['image']
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
        img = data['image']
        text = data['label']
        filename = str(i)+".jpg"
        local_path = dataset_path+"train/"
        local_img = local_path+filename
        if not os.path.exists(local_img):
            img.save(local_img)
        test_data[i] = [local_img, text]
    part = int(test_len/2)
    valid_data = test_data[:part]
    test_data = test_data[part:]
    return train_data, valid_data, test_data

if __name__ == '__main__':
    # read_coco()
    # read_flowers()
    # read_pokemon()
    # read_imagereward()
    read_cub200()
    # read_cifar10()
