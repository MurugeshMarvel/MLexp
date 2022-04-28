import os
import math

import torch as T
import hydra
from omegaconf import DictConfig

from src.models import CapsuleNet


@hydra.main(config_path="config/config.yaml")
def main(cfg):
    