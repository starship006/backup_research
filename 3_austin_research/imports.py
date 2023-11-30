# %%
import plotly
import plotly.express as px
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
import pickle

import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
from tqdm import tqdm
import random
from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader
from typing import Union, List, Optional, Callable, Tuple, Dict, Literal, Set
from jaxtyping import Float, Int
from jaxtyping import Float, Int, Bool, Shaped, jaxtyped
import typeguard
from functools import partial, wraps
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.utils import to_numpy
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache, patching

import circuitsvis as cv
import os, sys

import urllib.request
import zipfile

if not os.path.exists("path_patching.py"):
    urllib.request.urlretrieve("https://github.com/callummcdougall/path_patching/archive/refs/heads/main.zip", "main.zip")
    
    with zipfile.ZipFile("main.zip", "r") as zip_ref:
        zip_ref.extract("path_patching-main/ioi_dataset.py")
        zip_ref.extract("path_patching-main/path_patching.py")
    
    sys.path.append("path_patching-main")
    
    os.remove("main.zip")
    os.rename("path_patching-main/ioi_dataset.py", "ioi_dataset.py")
    os.rename("path_patching-main/path_patching.py", "path_patching.py")
    os.rmdir("path_patching-main")


# if not os.path.exists("path_patching.py"):
#         !wget https://github.com/callummcdougall/path_patching/archive/refs/heads/main.zip
#         !unzip main.zip 'path_patching-main/ioi_dataset.py'
#         !unzip main.zip 'path_patching-main/path_patching.py'
#         sys.path.append("path_patching-main")
#         os.remove("main.zip")
#         os.rename("path_patching-main/ioi_dataset.py", "ioi_dataset.py")
#         os.rename("path_patching-main/path_patching.py", "path_patching.py")
#         os.rmdir("path_patching-main")

from path_patching import Node, IterNode, path_patch, act_patch


from neel_plotly import imshow, line, scatter, histogram
torch.set_grad_enabled(False)
#device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
#device

# %%
update_layout_set = {
    "xaxis_range", "yaxis_range", "hovermode", "xaxis_title", "yaxis_title", "colorbar", "colorscale", "coloraxis", "title_x", "bargap", "bargroupgap", "xaxis_tickformat",
    "yaxis_tickformat", "title_y", "legend_title_text", "xaxis_showgrid", "xaxis_gridwidth", "xaxis_gridcolor", "yaxis_showgrid", "yaxis_gridwidth", "yaxis_gridcolor",
    "showlegend", "xaxis_tickmode", "yaxis_tickmode", "xaxis_tickangle", "yaxis_tickangle", "margin", "xaxis_visible", "yaxis_visible", "bargap", "bargroupgap"
}

def imshow(tensor, return_fig = False, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    facet_labels = kwargs_pre.pop("facet_labels", None)
    border = kwargs_pre.pop("border", False)
    if "color_continuous_scale" not in kwargs_pre:
        kwargs_pre["color_continuous_scale"] = "RdBu"
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    if facet_labels:
        for i, label in enumerate(facet_labels):
            fig.layout.annotations[i]['text'] = label
    if border:
        fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    # things like `xaxis_tickmode` should be applied to all subplots. This is super janky lol but I'm under time pressure
    for setting in ["tickangle"]:
      if f"xaxis_{setting}" in kwargs_post:
          i = 2
          while f"xaxis{i}" in fig["layout"]:
            kwargs_post[f"xaxis{i}_{setting}"] = kwargs_post[f"xaxis_{setting}"]
            i += 1
    fig.update_layout(**kwargs_post)
    
    if return_fig:
      return fig
    else:
      fig.show(renderer=renderer)

def hist(tensor, renderer=None, **kwargs):
    kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
    kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
    names = kwargs_pre.pop("names", None)
    if "barmode" not in kwargs_post:
        kwargs_post["barmode"] = "overlay"
    if "bargap" not in kwargs_post:
        kwargs_post["bargap"] = 0.0
    if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
        kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
    fig = px.histogram(x=tensor, **kwargs_pre).update_layout(**kwargs_post)
    if names is not None:
        for i in range(len(fig.data)):
            fig.data[i]["name"] = names[i // 2]
    fig.show(renderer)


from plotly import graph_objects as go
from plotly.subplots import make_subplots

import argparse