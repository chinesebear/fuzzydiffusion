<!-- <p align="center">
<img src="./doc/png/logo-transparent.png" width=20% />
</p> -->
<h1 align="center">
Diffusion Fuzzy System: Fuzzy Rule Guided Latent Multi-Path Diffusion Modeling
</h1>
<p align="center">
<img src="https://img.shields.io/badge/OS-Ubuntu22.4-blue" />
<img src="https://img.shields.io/badge/Python-3.8-red" />
<img src="https://img.shields.io/badge/Torch-2.3.0-red" />
<img src="https://img.shields.io/badge/Build-Success-green" />
<img src="https://img.shields.io/badge/License-BSD-blue" />
<img src="https://img.shields.io/badge/Release-0.1-blue" />
</p>

<p align="center">
<img src="doc/png/fig.dfs.png" width=100% /> <br>
<b>Figure 1</b>: The structure of DFS.
</p>

<p align="justify">
Diffusion models have emerged as a leading technique for generating images due to their ability to create high-resolution and realistic images. Despite their strong performance, diffusion models still struggle in managing image collections with significant feature differences. They often fail to capture complex features and produce conflicting results. Research has attempted to address this issue by learning different regions of an image through multiple diffusion paths and then combining them. However, this approach leads to inefficient coordination among multiple paths and high computational costs. To tackle these issues, this paper presents a Diffusion Fuzzy System (DFS), a latent-space multi-path diffusion model guided by fuzzy rules. DFS offers several advantages. First, unlike traditional multi-path diffusion methods, DFS uses multiple diffusion paths, each dedicated to learning a specific class of image features. By assigning each path to a different feature type, DFS overcomes the limitations of multi-path models in capturing heterogeneous image features. Second, DFS employs rule-chain-based reasoning to dynamically steer the diffusion process and enable efficient coordination among multiple paths. Finally, DFS introduces a fuzzy membership-based latent-space compression mechanism to reduce the computational costs of multi-path diffusion effectively. We tested our method on three public datasets: LSUN Bedroom, LSUN Church, and MS COCO. The results show that DFS achieves more stable training and faster convergence than existing single-path and multi-path diffusion models. Additionally, DFS surpasses baseline models in both image quality and alignment between text and images, and also shows improved accuracy when comparing generated images to target references.
</p>

## Environment Setup

> Our project relies on [LDM](https://github.com/CompVis/latent-diffusion) project.

Tested with **Python 3.8.5** on Linux x86_64 (CUDA 12.1 runtime wheels).

``` shell
conda create -n fuzzydiffusion python=3.8.5 pip -y
conda activate fuzzydiffusion
pip install -r requirements.txt
```

`requirements.txt` pins the PyPI dependencies and installs the three source
packages (`latent-diffusion`, `taming-transformers`, `CLIP`). See
[Important Dependencies](#important-dependencies) for compatibility notes.

## Datasets

### LSUN
<p align="justify">
The Large-scale Scene Understanding (LSUN) challenge aims to provide a different benchmark for large-scale scene classification and understanding. The LSUN classification dataset contains 10 scene categories, such as dining room, bedroom, chicken, outdoor church, and so on. For training data, each category contains a huge number of images, ranging from around 120,000 to 3,000,000. The validation data includes 300 images, and the test data has 1000 images for each category. https://github.com/fyu/lsun
</p>


To facilitate the experiments, we examined the LSUN Bedroom and LSUN Church datasets and provided [download link](https://gofile.me/7y6ht/61yKRTxNi) for zip (92GB, `unzip lsun.zip`) . After downloading, place the `lsun` folder into the `datasets` directory located at the root of the project, as shown below:

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

The loader follows the paper's **7:1:2** split: each `<subset>_train.txt` list is
sliced deterministically into train / validation / test (70% / 10% / 20%) and
all splits read images from `<subset>_train/`. Evaluation always uses the
`test` slice.

### MS_COCO
<p align="justify">
The MS COCO (Microsoft Common Objects in Context) dataset is a large-scale object detection, segmentation, key-point detection, and captioning dataset. The dataset consists of 328K images. 

HuggingFace: https://huggingface.co/datasets/ChristophSchuhmann/MS_COCO_2017_URL_TEXT

The 591,753-image `train` split is divided 7:1:2 following the paper. Images are
downloaded lazily from their URLs on first access in `loader.py`, so no manual
work is required (the first epoch is network-bound).
</p>

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

Representatives (one per rule chain) are selected with **K-Medoids** in the SMC
feature space (BLIP+BERT semantic features and VGG16 features), following
Eq. 9 of the paper.

```shell
# LSUN Church / Bedroom (alpha = 0.3)
python3 fcm.py --dataset lsun_church  --rules 2 3 4 5 6 7 8 9 10
python3 fcm.py --dataset lsun_bedroom --rules 2 3 4 5 6 7 8 9 10

# MS COCO / CUB-200 (alpha = 0.7)
python3 fcm.py --dataset coco  --rules 3 --alpha 0.7
python3 fcm.py --dataset cub200 --rules 3 --alpha 0.7
```

Outputs are written to `output/delegates/<dataset>/` as images plus a
`<dataset>_<K>_delegates.csv` file consumed by the training scripts.
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
           f"{options.base_path}src/config/latent-diffusion/lsun_churches-fuzzy-ldm-kl-8.yaml")
lsun_train('bedrooms',
           f"{options.base_path}src/config/latent-diffusion/lsun_bedrooms-fuzzy-ldm-vq-4.yaml")
```

The frozen LDM / first-stage checkpoints referenced by `ckpt_path` in the
configs must be placed under `output/models/` (or the paths updated in the
`*fuzzy*.yaml` files).

<p align="center">
<img src="doc/fig.convergence.jpg" width=100% /> <br>
<b>Figure 2</b>: The convegence curve of DFS
</p>


The testing program is automatically triggered upon the completion of training.
```python
logger.info("lsun test start")
lsun_test(dataset_name, fldmodel, root_path, ddim_steps=1000)
logger.info("lsun test done")
logger.remove(log_file)
```

train model on MS_COCO dataset through `coco.py`, rule number default `3`.

```shell
python3 coco.py
```

```python
coco_train(f"{options.base_path}src/config/latent-diffusion/coco-fuzzy-ldm-vq-4.yaml")
```

The testing program is automatically triggered upon the completion of training.

```python
logger.info("coco test start")
coco_test(fldmodel, root_path, ddim_steps=1000)
logger.info("coco test done")
logger.remove(log_file)
```

<p align="center">
<img src="doc/png/fig.txt2img.png" width=100% /> <br>
<b>Figure 3</b>: Images generated by DFS with the specified
prompts.
</p>

## Important Dependencies

The environment is Python **3.8.5** / Linux x86_64. `requirements.txt` is the
single source of truth; the versions below are the compatibility-critical ones
and should not be changed casually.

| Component | Pinned version | Why it matters |
|---|---|---|
| `torch` / `torchvision` | `2.3.0` / `0.18.0` | CUDA 12.1 wheels; `nvidia-*-cu12` and `triton` are pulled automatically. `torchvision 0.18.0` is the matching build (the old conda `torchvision 0.8.1` cannot import with torch 2.3). |
| `numpy` | `1.24.4` | `scikit-image 0.21` and `datasets 2.18` require numpy >= 1.21; the old export's `1.20.3` is incompatible. |
| `lightning` / `pytorch-lightning` | `1.9.5` | The code needs **both** `pytorch_lightning.utilities.distributed.rank_zero_only` (LDM + `src/ldmodel.py`) and `lightning.pytorch.loggers`; only the 1.9.x pair provides both. |
| `lightning-cloud` | `0.5.70` | Must stay `< 0.6`; `lightning-cloud 0.6.x` breaks `import lightning` for Lightning 1.9.5. |
| `transformers` / `tokenizers` | `4.46.3` / `0.20.3` | BLIP captioning and BERT embeddings used by SMC; tokenizers must stay in the `0.20.x` range. |
| `torchmetrics` / `torch-fidelity` | `1.5.2` / `0.3.0` | FID, MIFID, IS and KID; `torch-fidelity` supplies the Inception backbone used by torchmetrics. |
| `diffusers` / `datasets` | `0.30.3` / `2.18.0` | APIs used by the training/evaluation scripts. |
| `scikit-image` | `0.21.0` | `skimage.metrics.structural_similarity` used in the training code and baselines. |
| `latent-diffusion` | CompVis `main` | Provides the `ldm` package. Not on PyPI; installed from GitHub by `requirements.txt`. |
| `taming-transformers` | CompVis `master` | Required by `ldm.models.autoencoder` (VQ models). Not on PyPI. |
| `clip` | OpenAI CLIP | `ldm.modules.encoders.modules` imports `clip` at module load, so the COCO `BERTEmbedder` path needs it. Installed from GitHub by `requirements.txt`. |

**Model weights** (not pip packages; downloaded on first use, then cached):

| Weight | Size (approx.) | Cache |
|---|---|---|
| `Salesforce/blip-image-captioning-base` | 991 MB | `~/.cache/huggingface/hub` |
| `bert-base-uncased` (safetensors) | 441 MB | `~/.cache/huggingface/hub` |
| torchvision VGG16 `IMAGENET1K_V1` | 553 MB | `~/.cache/torch/hub/checkpoints` |

If the machine is offline, pre-populate those two caches and run with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

The legacy conda export is intentionally **not** used: its
`tensorflow 2.13` / `keras` / `scann` pins conflict with `torch 2.3`
(`typing-extensions`) and `numpy 1.24`, and the conda `pytorch 1.7` /
`cudatoolkit 11.0` packages are superseded by the pip CUDA 12.1 wheels.




