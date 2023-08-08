# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import numpy as np
import pandas as pd
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
# import plotly.express as px
from torch.utils.data import DataLoader
from typing import Union, List, Optional, Callable, Tuple, Dict, Literal, Set
from jaxtyping import Float, Int
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.utils import to_numpy
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache, patching

import plotly.express as px
import circuitsvis as cv
import os, sys
import plotly


# %%
from neel_plotly import imshow, line, scatter, histogram
import tqdm
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
device

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
    fig.show(renderer=renderer)
    if return_fig:
      return fig

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

# %%
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# %%

model_name = "gpt-j-6B"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    #refactor_factored_attn_matrices=True,
    device = device,
)
# %%

start_prompt = "If spelling the string \" table\" in all capital letters separated by hyphens gives T-A-B-L-E then spelling the string \" tree\" in all capital letters, separated by hyphens, gives "
print(model.to_str_tokens(start_prompt))

for i in range(10):
    logits = model(start_prompt)
    next_token = torch.argmax(logits[0, -1, :], dim=-1)
    start_prompt +=  model.to_single_str_token(next_token.item())

print(start_prompt)
# %%

# generate prompts for word list
word_list = [
    "apple", "banana", "orange", "grape", "mango",
    "strawberry", "blueberry", "watermelon", "kiwi", "pineapple",
    "peach", "pear", "cherry", "plum", "lemon", "the", "code", "will", "print", "list", "containing", "randomly", "words", "from", "want",
    "for", "cat", "dog", "par", "pat", "tea", "tip", "pit", "pet", "tik", "tok"
]

prompts = []
last_letters = []

for word in word_list:
    #    prompt = f"If spelling the string \" table\" in all capital letters separated by hyphens gives T-A-B-L-E then spelling the string \"{word}\" in all capital letters, separated by hyphens, gives "

    prompt = f"How do you spell \"{word}\"? "
    for letter in word:
        # add letters in caps
        prompt += f"{letter.upper()}-"
    
    prompts.append(prompt[:-2])
    last_letters.append(prompt[-2])

# %%
print(prompts)
#last_letters

# %%
num_correct = 0
for prompt, last_letter in zip(prompts, last_letters):
    logits = model(prompt)
    next_token = torch.argmax(logits[0, -1, :], dim=-1)
    predicted_letter = model.to_single_str_token(next_token.item())
    if predicted_letter == last_letter:
        num_correct += 1
        print(f"Correctly predicted {last_letter} for {prompt}")
    else:
        print(f"Incorrectly predicted {predicted_letter} for {prompt}")

print(f"Accuracy: {num_correct / len(prompts)}")
# %%
model.cfg
# %%
utils.test_prompt(start_prompt, "T", model)

# %%
