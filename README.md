<!-- <p align="center">
<img src="./doc/png/logo-transparent.png" width=20% />
</p> -->
<h1 align="center">
RDM: Rule Guided Latent Space Multi-Path Diffusion Model
</h1>
<p align="center">
<img src="https://img.shields.io/badge/OS-Ubuntu22.4-blue" />
<img src="https://img.shields.io/badge/Python-3.8-red" />
<img src="https://img.shields.io/badge/Torch-1.8.0-red" />
<img src="https://img.shields.io/badge/Build-Success-green" />
<img src="https://img.shields.io/badge/License-BSD-blue" />
<img src="https://img.shields.io/badge/Release-0.1-blue" />
</p>

<p align="center">
<img src="doc/png/fig.structure.png" width=100% /> <br>
<b>Figure 1</b>: The structure of the proposed model (RDM).
</p>

## Conda Enviroment Setup

``` shell
conda create --name rdm --file ./requirements.txt
conda activate rdm
```

## Datasets

### LSUN
The Large-scale Scene Understanding (LSUN) challenge aims to provide a different benchmark for large-scale scene classification and understanding. The LSUN classification dataset contains 10 scene categories, such as dining room, bedroom, chicken, outdoor church, and so on. For training data, each category contains a huge number of images, ranging from around 120,000 to 3,000,000. The validation data includes 300 images, and the test data has 1000 images for each category. https://github.com/fyu/lsun

To facilitate the experiments, we examined the LSUN Bedroom and LSUN Church datasets and provided [download links](https://gofile.me/7y6ht/61yKRTxNi) for zip (92GB, `unzip lsun.zip`) .
After downloading, place the `lsun` folder into the `datasets` directory located at the root of the project, as shown below:
```shell
xxx@xx:~/xxx/github/fuzzydiffusion/datasets/lsun$ ll
drwxrwxr-x  8 yang yang      4096 Nov 24  2023 ./
drwxrwxr-x 13 yang yang      4096 Apr 25 11:28 ../
drwxrwxr-x  2 yang yang 248651776 Nov 22  2023 bedrooms_train/
-rw-rw-r--  1 yang yang 139289932 Sep 25  2023 bedrooms_train.txt
drwxrwxr-x  2 yang yang     32768 Nov 21  2023 bedrooms_val/
-rw-rw-r--  1 yang yang    230000 Nov 21  2023 bedrooms_val.txt
drwxrwxr-x  2 yang yang  10420224 Nov 21  2023 churches_train/
drwxrwxr-x  2 yang yang     28672 Nov 21  2023 churches_val/
-rw-rw-r--  1 yang yang   5576442 Sep 25  2023 church_outdoor_train.txt
-rw-rw-r--  1 yang yang    230000 Sep 25  2023 church_outdoor_val.txt
```

### MS_COCO

The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images. 

HuggingFace: https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT

MS_COCO is automatically downloaded in `loader.py`, so no manual work is required.
### Load Datasets

``` shell
cd src
python3 loader.py
```
log output:
``` log
2025-04-25 12:55:23.139 | INFO     | __main__:read_dataset:50 - ChristophSchuhmann/MS_COCO_2017_URL_TEXT- done
2025-04-25 12:55:23.140 | INFO     | __main__:data_preload:98 - ChristophSchuhmann/MS_COCO_2017_URL_TEXT data preload start
2025-04-25 12:55:23.145 | INFO     | __main__:read_dataset:50 - ChristophSchuhmann/MS_COCO_2017_URL_TEXT- done
data preload: 100%|████████████████████████████████| 591753/591753 [00:23<00:00, 25651.73it/s]
2025-04-25 12:55:46.218 | INFO     | __main__:data_preload:118 - ChristophSchuhmann/MS_COCO_2017_URL_TEXT data preload done
2025-04-25 12:55:46.218 | INFO     | __main__:data_preload:314 - LSUN/bedrooms-train data preload start
100%|████████████████████████████████| 3028042/3028042 [00:04<00:00, 671985.22it/s]
2025-04-25 12:55:50.884 | INFO     | __main__:data_preload:339 - LSUN/bedrooms-train data preload done
2025-04-25 12:55:50.940 | INFO     | __main__:data_preload:314 - LSUN/churches-train data preload start
100%|███████████████████████████████| 121227/121227 [00:00<00:00, 859921.34it/s]
2025-04-25 12:55:51.087 | INFO     | __main__:data_preload:339 - LSUN/churches-train data preload done
done
```
## Image Set Delegates For Rule Antecedents
To construct the rules, [2,...,10] representatives are selected.
```shell
python3 fcm.py
```
The representative selection for MS_COCO is as follows:
```shell
...
-rw-rw-r-- 1 yang yang  5556 Apr 25 15:31 coco_3_delegates_1.jpg
-rw-rw-r-- 1 yang yang  9585 Apr 25 15:31 coco_3_delegates_2.jpg
-rw-rw-r-- 1 yang yang 11917 Apr 25 15:31 coco_3_delegates_3.jpg
-rw-rw-r-- 1 yang yang   260 Apr 25 15:31 coco_3_delegates.csv
-rw-rw-r-- 1 yang yang 16955 Apr 25 15:31 coco_4_delegates_1.jpg
-rw-rw-r-- 1 yang yang  9757 Apr 25 15:31 coco_4_delegates_2.jpg
-rw-rw-r-- 1 yang yang  9157 Apr 25 15:31 coco_4_delegates_3.jpg
-rw-rw-r-- 1 yang yang  5556 Apr 25 15:31 coco_4_delegates_4.jpg
-rw-rw-r-- 1 yang yang   343 Apr 25 15:31 coco_4_delegates.csv
...
```
## Train and Test

train model on LSUN datasets through `lsun.py`, rule number default `3`.
```shell
python3 lsun.py
```
```python
lsun_train('churches', 
            f"{options.base_path}src/config/latent-diffusion/lsun_churches-ldm-kl-8.yaml",
            f"{options.base_path}output/delegates/lsun_church/lsun_church_3_delegates.csv",
            3)
lsun_train('bedrooms', 
            f"{options.base_path}src/config/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml",
            f"{options.base_path}output/delegates/lsun_bedroom/lsun_bedroom_3_delegates.csv",
            3)
```
The testing program is automatically triggered upon the completion of training.
```python
logger.info("lsun test start")
lsun_test(dataset_name, fldmodel, root_path)
logger.info("lsun test done")
logger.remove(log_file)
```

train model on MS_COCO dataset through `coco.py`, rule number default `3`.

```shell
python3 coco.py
```

```python
coco_train(f"{options.base_path}src/config/latent-diffusion/txt2img-1p4B-eval.yaml",
           f"{options.base_path}output/delegates/coco/coco_3_delegates.csv",
           3)
```

The testing program is automatically triggered upon the completion of training.

```python
logger.info("coco test start")
coco_test(fldmodel, root_path)
logger.info("coco test done")
logger.remove(log_file)
```

<p align="center">
<img src="doc/png/fig.txt2img.png" width=100% /> <br>
<b>Figure 2</b>: Images generated by RDM with the specified
prompts.
</p>


## Acknowledgement
Our project relies on [LDM](https://github.com/CompVis/latent-diffusion) project.
