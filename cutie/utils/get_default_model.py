"""
A helper function to get a default model for quick testing
"""
from omegaconf import open_dict
from hydra import compose, initialize

import torch
from cutie.model.cutie import CUTIE
from cutie.inference.utils.args_utils import get_dataset_cfg
from scripts.download_models import download_models_if_needed

import os, sys, omegaconf
import hydra

def get_default_model() -> CUTIE:
    # TODO this might be a hack with unintended side effects...
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base='1.3.2', config_path="../config", job_name="eval_config"):
        cfg = compose(config_name="eval_config")
    
        # # get current file path
        # current_path = os.path.dirname(os.path.realpath(__file__))
        # # get the relative path to the config file
        # config_file_path = os.path.join(current_path, "../config/eval_config.yaml")
        # cfg = omegaconf.OmegaConf.load(config_file_path)

        weights_root_dirpath=download_models_if_needed()
        with open_dict(cfg):
            cfg['weights'] = os.path.join(weights_root_dirpath, 'cutie-base-mega.pth')
        get_dataset_cfg(cfg)

        # Load the network weights
        cutie = CUTIE(cfg).cuda().eval()
        model_weights = torch.load(cfg.weights)
        cutie.load_weights(model_weights)

        return cutie
