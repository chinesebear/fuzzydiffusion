import time
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
from omegaconf import OmegaConf
import torch

from util import instantiate_from_config
from ddpm import disabled_train
from setting import options
from distributions import DiagonalGaussianDistribution

def latent_structural_similarityI(a, b , data_range=255, multichannel=True):
    score = 0.5
    return score


def instantiate_first_stage(config):
    model = instantiate_from_config(config)
    first_stage_model = model.eval()
    first_stage_model.train = disabled_train
    for param in first_stage_model.parameters():
        param.requires_grad = False
    return first_stage_model

def img_to_tensor(images):
    imgs = []
    for img in images:
        img = np.array(img)
        H, W, C = img.shape
        img = img.reshape((C, H, W))
        imgs.append(img)
    imgs = np.array(imgs)
    x_0 = torch.from_numpy(imgs).to(options.device).float()
    x_0 = x_0 / 127.5 - 1.0  #[-1.0, 1.0]
    return x_0

def get_first_stage_encoding(encoder_posterior):
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
    return z
    # return self.scale_factor * z

if __name__ == "__main__":
    src = Image.open('/home/yang/sda/github/fuzzydiffusion/doc/cat.jpg')
    tgt = Image.open('/home/yang/sda/github/fuzzydiffusion/doc/cat-1.jpg')
    src_gray = np.array(src.convert('L'))
    tgt_gray = np.array(tgt.convert('L'))
    (score, diff) = structural_similarity(src_gray, tgt_gray, win_size=101, full=True)

    src_rgb = np.array(src)
    tgt_rgb = np.array(tgt)
    (score1, diff1) = structural_similarity(src_rgb, tgt_rgb, win_size=101, channel_axis=-1, full=True)

    configs = OmegaConf.load("/home/yang/sda/github/fuzzydiffusion/src/config/latent-diffusion/lsun_churches-fuzzy-ldm-kl-8.yaml")
    print(configs.model.params.first_stage_config)
    model = instantiate_first_stage(configs.model.params.first_stage_config).to(options.device)
    print("ok")
    x = img_to_tensor([src])
    encoder_posterior = model.encode(x)
    z = get_first_stage_encoding(encoder_posterior).detach()
    print(z.shape)