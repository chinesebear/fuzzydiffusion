"""DFS on CelebA-HQ (not part of the paper's main experiments).

Kept as an additional unconditional dataset following the same training and
generation code paths (Algorithm 1/2).
"""

import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from loguru import logger
from tqdm import tqdm

from ldm.util import instantiate_from_config

from loader import CELEBA_HQ
from setting import options
from metrics import Evaluator
from fldm import FuzzyLatentDiffusion, load_delegates
from utils import setup_seed, save_model, frozen_model, stat_model_param, \
    csv_dist_record, metrics_mean, csv_record, metrics_fig, combine_imgs3, \
    check_dir, load_config, resolve_path

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def celeba_hq_train(config_path, delegate_path=None, rule_num=None, epoch=20):
    setup_seed(options.split_seed)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path + f'output/log/celeba_hq/{now}/'
    model_path = options.base_path + "output/pth/"
    check_dir(root_path)
    check_dir(model_path)
    log_file = logger.add(
        root_path + 'fuzzy-latent-diffusion-train-' + str(datetime.date.today()) + '.log')

    config = load_config(config_path)
    fuzzy = config.get("fuzzy", {})
    rule_num = int(rule_num or fuzzy.get("rule_num", options.rule_num))
    delegate_path = resolve_path(delegate_path or fuzzy.get("delegate_path"))
    if delegate_path is None:
        raise ValueError("no delegate CSV: pass delegate_path")

    model_config = config.pop("model", None)
    ldmodel = instantiate_from_config(model_config).to(options.device)
    fldmodel = FuzzyLatentDiffusion(
        ldmodel,
        rule_num=rule_num,
        delegates=load_delegates(delegate_path)[:rule_num],
        alpha=float(fuzzy.get("alpha", 0.3)),
        beta_1=float(fuzzy.get("beta_1", 0.0015)),
        beta_T=float(fuzzy.get("beta_T", 0.0155)),
        T=int(fuzzy.get("T", 1000)),
        root_path=root_path,
        condition=False,
        smc_every=int(fuzzy.get("smc_every", options.smc_every)),
    ).to(options.device)
    frozen_model(fldmodel)
    stat_model_param(fldmodel, "fldmodel")

    dataset = CELEBA_HQ(options.base_path + 'datasets/celeba_hq_256',
                        phase='train', transform=TRANSFORM)
    train_data = DataLoader(dataset, batch_size=32, num_workers=5, shuffle=True,
                            drop_last=True, pin_memory=True)

    checkpoint = model_path + f"fldm_celeba_hq_epoch_{epoch}_rules_{rule_num}.pt"
    if not os.path.exists(checkpoint):
        fldmodel.fuzzy_trainer(train_data, epoch)
        save_model(fldmodel, checkpoint)

    celeba_hq_test(fldmodel, root_path,
                   ddim_steps=int(fuzzy.get("ddim_steps", options.ddim_steps)))
    logger.remove(log_file)


def celeba_hq_test(fldmodel, root_path, ddim_steps=None):
    fldmodel.eval()
    frozen_model(fldmodel)
    dist_csv_path = root_path + 'distribution-' + str(datetime.date.today()) + '.csv'
    metrics_csv_path = (
        root_path + f'celeba_hq_{fldmodel.rule_num}rules_metrics-'
        + str(datetime.date.today()) + '.csv')

    evaluator = Evaluator(torch.device("cpu"))
    evaluator.reset()
    dataset = CELEBA_HQ(options.base_path + 'datasets/celeba_hq_256',
                        phase='test', transform=TRANSFORM)
    test_data = DataLoader(dataset, batch_size=32, num_workers=5, shuffle=True,
                           drop_last=True, pin_memory=True)

    sigmoid = nn.Sigmoid()
    met_data = {key: [] for key in
                ['fid', 'is', 'mifid', 'kid', 'psnr', 'ms_ssim', 'ssim',
                 'precision', 'recall']}

    for _ in tqdm(range(200)):
        batch = next(iter(test_data))
        x = batch['image'].cuda()
        x_ref, _, x_fuzz, _ = fldmodel(batch, latent_output=True, ddim_steps=ddim_steps)
        z = fldmodel.ldmodel.get_input(batch, 'image')[0]
        x_p = fldmodel.ldmodel.decode_first_stage(z)

        x_norm, x_p_norm, x_fuzz_norm = sigmoid(x_ref), sigmoid(x_p), sigmoid(x_fuzz)
        csv_dist_record(dist_csv_path, {
            'x': x, 'z': z, 'x_p': x_p, 'x_fuzz': x_fuzz,
            'x_norm': x_norm, 'x_p_norm': x_p_norm, 'x_fuzz_norm': x_fuzz_norm})
        combine_imgs3(x_norm, x_p_norm, x_fuzz_norm)

        real_imgs, gen_imgs = x_norm.cpu(), x_fuzz_norm.cpu()
        fid = evaluator.calc_fid(real_imgs, gen_imgs)
        _is = evaluator.calc_is(gen_imgs)
        mifid = evaluator.calc_mifid(real_imgs, gen_imgs)
        kid = evaluator.calc_kid(real_imgs, gen_imgs)
        met_data['fid'].append(fid)
        met_data['is'].append(_is)
        met_data['mifid'].append(mifid)
        met_data['kid'].append(kid)

        met_data['psnr'].append(evaluator.calc_psnr(real_imgs, gen_imgs))
        met_data['ms_ssim'].append(evaluator.calc_ms_ssim(real_imgs, gen_imgs))
        met_data['ssim'].append(evaluator.calc_ssim(real_imgs, gen_imgs))
        precision, recall = evaluator.calc_preision_recall(real_imgs, gen_imgs)
        met_data['precision'].append(precision)
        met_data['recall'].append(recall)

        csv_record(metrics_csv_path, {
            'fid': str(fid), 'is': str(_is), 'mifid': str(mifid), 'kid': str(kid),
            'psnr': str(metrics_mean(met_data['psnr'])),
            'ms_ssim': str(metrics_mean(met_data['ms_ssim'])),
            'ssim': str(metrics_mean(met_data['ssim'])),
            'precision': str(metrics_mean(met_data['precision'])),
            'recall': str(metrics_mean(met_data['recall'])),
        })
        metrics_fig("fid", met_data['fid'])


if __name__ == '__main__':
    celeba_hq_train(
        options.base_path + "src/config/latent-diffusion/lsun_churches-fuzzy-ldm-kl-8.yaml",
        delegate_path=options.base_path + "output/delegates/celeba_hq/celeba_hq_3_delegates.csv",
        rule_num=3,
    )
