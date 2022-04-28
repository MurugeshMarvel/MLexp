import torch as T
from torch import nn
import sys

sys.path.append('../')
from src.models import SwinTransformer

model_kwargs = dict(
    img_size=224, in_chans=1,
    patch_size=2, window_size=2, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))

model = SwinTransformer(**model_kwargs)

inp = T.rand((1,1,224, 224))

out = model.forward(inp)