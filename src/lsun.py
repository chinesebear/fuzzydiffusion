import datetime
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from loguru import logger
from tqdm import tqdm

from ldm.util import instantiate_from_config

from loader import LSUN
from setting import options
from metrics import Evaluator
from fldm import FuzzyLatentDiffusion, load_delegates
from utils import setup_seed, save_model, frozen_model, stat_model_param, \
    csv_dist_record, metrics_mean, csv_record, metrics_fig, \
    combine_imgs3, check_dir, load_config, resolve_path

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def _fuzzy_settings(config, default_delegate=None):
    fuzzy = config.get("fuzzy", {})
    return {
        "rule_num": int(fuzzy.get("rule_num", options.rule_num)),
        "alpha": float(fuzzy.get("alpha", 0.3)),
        "beta_1": float(fuzzy.get("beta_1", 0.0015)),
        "beta_T": float(fuzzy.get("beta_T", 0.0155)),
        "T": int(fuzzy.get("T", 1000)),
        "ddim_steps": int(fuzzy.get("ddim_steps", options.ddim_steps)),
        "cfg_scale": float(fuzzy.get("cfg_scale", 1.0)),
        "smc_every": int(fuzzy.get("smc_every", options.smc_every)),
        "delegate_path": resolve_path(fuzzy.get("delegate_path", default_delegate)),
    }


def lsun_train(dataset_name, config_path, delegate_path=None, rule_num=None, epoch=20):
    setup_seed(options.split_seed)
    batch_size = 32

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    root_path = options.base_path + f'output/log/lsun_{dataset_name}/{now}/'
    model_path = options.base_path + "output/pth/"
    check_dir(root_path)
    check_dir(model_path)
    log_file = logger.add(
        root_path + 'fuzzy-latent-diffusion-train-' + str(datetime.date.today()) + '.log'
    )

    config = load_config(config_path)
    fuzzy = _fuzzy_settings(config, delegate_path)
    if rule_num is not None:
        fuzzy["rule_num"] = int(rule_num)
    if fuzzy["delegate_path"] is None:
        raise ValueError("no delegate CSV: set `fuzzy.delegate_path` or pass one")

    model_config = config.pop("model", None)
    ldmodel = instantiate_from_config(model_config).to(options.device)
    delegates = load_delegates(fuzzy["delegate_path"])
    if len(delegates) != fuzzy["rule_num"]:
        logger.warning(
            f"{len(delegates)} delegates but rule_num={fuzzy['rule_num']}; "
            "the first K are used"
        )
    fldmodel = FuzzyLatentDiffusion(
        ldmodel,
        rule_num=fuzzy["rule_num"],
        delegates=delegates[:fuzzy["rule_num"]],
        alpha=fuzzy["alpha"],
        beta_1=fuzzy["beta_1"],
        beta_T=fuzzy["beta_T"],
        T=fuzzy["T"],
        root_path=root_path,
        condition=False,
        cfg_scale=fuzzy["cfg_scale"],
        smc_every=fuzzy["smc_every"],
    ).to(options.device)
    frozen_model(fldmodel)
    stat_model_param(fldmodel, "fldmodel")

    dataset = LSUN(options.base_path + 'datasets/lsun',
                   dataset_sub_path=dataset_name,
                   phase='train',
                   transform=TRANSFORM)
    train_data = DataLoader(
        dataset, batch_size=batch_size, num_workers=5, shuffle=True,
        drop_last=True, pin_memory=True)

    checkpoint = model_path + f"fldm_lsun_{dataset_name}_epoch_{epoch}_rules_{fuzzy['rule_num']}.pt"
    if not os.path.exists(checkpoint):
        fldmodel.fuzzy_trainer(train_data, epoch)
        save_model(fldmodel, checkpoint)

    logger.info("lsun test start")
    lsun_test(dataset_name, fldmodel, root_path, ddim_steps=fuzzy["ddim_steps"])
    logger.info("lsun test done")
    logger.remove(log_file)


def lsun_test(dataset_name, fldmodel, root_path, ddim_steps=None):
    fldmodel.eval()
    frozen_model(fldmodel)
    log_file = logger.add(
        root_path + 'fuzzy-latent-diffusion-test-' + str(datetime.date.today()) + '.log'
    )
    metrics_csv_path = (
        root_path + f'lsun_{dataset_name}_{fldmodel.rule_num}rules_metrics-'
        + str(datetime.date.today()) + '.csv'
    )
    dist_csv_path = root_path + 'distribution-' + str(datetime.date.today()) + '.csv'

    evaluator = Evaluator(torch.device("cpu"))
    evaluator.reset()
    dataset = LSUN(options.base_path + 'datasets/lsun',
                   dataset_sub_path=dataset_name,
                   phase='test',
                   transform=TRANSFORM)
    batch_size = 32
    test_data = DataLoader(
        dataset, batch_size=batch_size, num_workers=5, shuffle=True,
        drop_last=True, pin_memory=True)

    sigmoid = nn.Sigmoid()
    met_data = {key: [] for key in
                ['fid', 'is', 'mifid', 'kid', 'psnr', 'ms_ssim', 'ssim',
                 'precision', 'recall']}

    for _ in tqdm(range(200)):  # 200 * 32 = 6400 evaluation images
        batch = next(iter(test_data))
        x = batch['image'].cuda()

        x_ref, _, x_fuzz, _ = fldmodel(
            batch, latent_output=True, ddim_steps=ddim_steps)

        z = fldmodel.ldmodel.get_input(batch, 'image')[0]
        x_p = fldmodel.ldmodel.decode_first_stage(z)

        x_norm = sigmoid(x_ref)
        x_p_norm = sigmoid(x_p)
        x_fuzz_norm = sigmoid(x_fuzz)

        distributions = {
            'x': x, 'z': z, 'x_p': x_p, 'x_fuzz': x_fuzz,
            'x_norm': x_norm, 'x_p_norm': x_p_norm, 'x_fuzz_norm': x_fuzz_norm,
        }
        csv_dist_record(dist_csv_path, distributions)
        combine_imgs3(x_norm, x_p_norm, x_fuzz_norm)

        # Paper: generated images are compared with the target references.
        real_imgs = x_norm.cpu()
        gen_imgs = x_fuzz_norm.cpu()
        fid = evaluator.calc_fid(real_imgs, gen_imgs)
        _is = evaluator.calc_is(gen_imgs)
        mifid = evaluator.calc_mifid(real_imgs, gen_imgs)
        kid = evaluator.calc_kid(real_imgs, gen_imgs)
        met_data['fid'].append(fid)
        met_data['is'].append(_is)
        met_data['mifid'].append(mifid)
        met_data['kid'].append(kid)
        logger.info(f"fid:{fid},is:{_is},mifid:{mifid},kid:{kid}")

        psnr = evaluator.calc_psnr(real_imgs, gen_imgs)
        ms_ssim = evaluator.calc_ms_ssim(real_imgs, gen_imgs)
        ssim = evaluator.calc_ssim(real_imgs, gen_imgs)
        met_data['psnr'].append(psnr)
        met_data['ms_ssim'].append(ms_ssim)
        met_data['ssim'].append(ssim)
        logger.info(
            f"psnr:{metrics_mean(met_data['psnr'])},"
            f"ms_ssim:{metrics_mean(met_data['ms_ssim'])},"
            f"ssim:{metrics_mean(met_data['ssim'])}")

        precision, recall = evaluator.calc_preision_recall(real_imgs, gen_imgs)
        met_data['precision'].append(precision)
        met_data['recall'].append(recall)
        logger.info(
            f"precision:{metrics_mean(met_data['precision'])},"
            f"recall:{metrics_mean(met_data['recall'])}")

        record_row = {
            'fid': str(fid),
            'is': str(_is),
            'mifid': str(mifid),
            'kid': str(kid),
            'psnr': str(metrics_mean(met_data['psnr'])),
            'ms_ssim': str(metrics_mean(met_data['ms_ssim'])),
            'ssim': str(metrics_mean(met_data['ssim'])),
            'precision': str(metrics_mean(met_data['precision'])),
            'recall': str(metrics_mean(met_data['recall'])),
        }
        csv_record(metrics_csv_path, record_row)

        metrics_fig("fid", met_data['fid'])
        metrics_fig("precision", met_data['precision'])
        metrics_fig("recall", met_data['recall'])
        metrics_fig("psnr", met_data['psnr'])

    logger.remove(log_file)


if __name__ == '__main__':
    lsun_train(
        'churches',
        options.base_path + "src/config/latent-diffusion/lsun_churches-fuzzy-ldm-kl-8.yaml",
    )
    lsun_train(
        'bedrooms',
        options.base_path + "src/config/latent-diffusion/lsun_bedrooms-fuzzy-ldm-vq-4.yaml",
    )
