"""i realize that i'm largely using the same functions over and over again. i want a single place to keep all of them

this is that place"""


# Imports
#%pip install git+https://github.com/neelnanda-io/TransformerLens.git
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
#%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
import circuitsvis as cv
import os, sys

if not os.path.exists("path_patching.py"):
        !wget https://github.com/callummcdougall/path_patching/archive/refs/heads/main.zip
        !unzip /content/main.zip 'path_patching-main/ioi_dataset.py'
        !unzip /content/main.zip 'path_patching-main/path_patching.py'
        sys.path.append("/content/path_patching-main")
        os.remove("/content/main.zip")
        os.rename("/content/path_patching-main/ioi_dataset.py", "ioi_dataset.py")
        os.rename("/content/path_patching-main/path_patching.py", "path_patching.py")
        os.rmdir("/content/path_patching-main")

from path_patching import Node, IterNode, path_patch, act_patch

#%pip install git+https://github.com/neelnanda-io/neel-plotly.git
from neel_plotly import imshow, line, scatter, histogram
import tqdm
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"

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



from plotly import graph_objects as go
from plotly.subplots import make_subplots



# Functions
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"],
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.

    logits: logits to get logit diff from
    answer_tokens: correct and incorrect answer tokens
    per_prompt: whether to return the array of differences rather than the average
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens.to(device))
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()



def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache: ActivationCache,
    logit_diff_directions: Float[Tensor, "batch d_model"],
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream. Applies LN to Residual Stack before
    getting the logit diff.

    residual_stack: stack of components in the residual stream to get logit diff from
    cache: cache of activations from the model (needed for LN)
    logit_diff_directions: directions to take the logit difference in
    '''
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)

    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size



def calc_all_logit_diffs(cache, model, correct_residual_direction, incorrect_residual_direction):
  """
  Calculate the Logit Diff, Correct Logit Output, and Incorrect Logit Output across all heads of the model
  """
  clean_per_head_residual, labels = cache.stack_head_results(layer = -1, return_labels = True, apply_ln = False) # per_head_residual.shape = heads batch seq_pos d_model
  # also, for the worried, no, we're not missing the application of LN here since it gets applied in the below function call
  per_head_logit_diff: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, correct_residual_direction - incorrect_residual_direction)

  per_head_logit_diff = einops.rearrange(
      per_head_logit_diff,
      "(layer head) ... -> layer head ...",
      layer = model.cfg.n_layers
  )

  correct_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = correct_residual_direction)

  correct_direction_per_head_logit = einops.rearrange(
      correct_direction_per_head_logit,
      "(layer head) ... -> layer head ...",
      layer = model.cfg.n_layers
  )

  incorrect_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = incorrect_residual_direction)

  incorrect_direction_per_head_logit = einops.rearrange(
      incorrect_direction_per_head_logit,
      "(layer head) ... -> layer head ...",
      layer = model.cfg.n_layers
  )

  return per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit

def display_all_logits(cache, model, correct_residual_direction, incorrect_residual_direction, answer_tokens,
                       title = "Logit Contributions", comparison = False, return_fig = False, logits = None, clean_cache = None):
    """
    Display the Logit Contributions across all heads of the model

    cache: cache of activations from the model
    title: title of the plot
    comparison: whether to also display differences between current and clean run
    return_fig: whether to return the figure
    logits: logits of current run (to display)
    """
    a,b,c = calc_all_logit_diffs(cache, model, correct_residual_direction, incorrect_residual_direction)
    if logits is not None:
        ld = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt = False)
    else:
        ld = 0.00

    if not comparison:
        fig = imshow(
            torch.stack([a,b,c]),
            return_fig = True,
            facet_col = 0,
            facet_labels = [f"Logit Diff - {ld:.2f}", "Correct Direction", "Incorrect Direction"],
            title=title,
            labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
            #coloraxis=dict(colorbar_ticksuffix = "%"),
            border=True,
            width=1500,
            margin={"r": 100, "l": 100}
        )
    else:

        ca, cb, cc = calc_all_logit_diffs(clean_cache, model, correct_residual_direction, incorrect_residual_direction)
        fig = imshow(
            torch.stack([a, b, c, a - ca, b - cb, c - cc]),
            return_fig = True,
            facet_col = 0,
            facet_labels = [f"Logit Diff - {ld:.2f}", "Correct Direction", "Incorrect Direction", "Logit Diff Diff", "Correction Direction Diff", "Incorrect Direction Diff"],
            title=title,
            labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
            #coloraxis=dict(colorbar_ticksuffix = "%"),
            border=True,
            width=1500,
            margin={"r": 100, "l": 100}
        )


    if return_fig:
        return fig


def stare_at_attention_and_head_pat(cache, layer_to_stare_at, head_to_isolate, model, clean_tokens, corrupted_tokens, display_corrupted_text = False, verbose = True, specific = False, specific_index = 0):
  """
  given a cache from a run, displays the attention patterns of a layer, as well as printing out how much the model
  """

  tokenized_str_tokens = model.to_str_tokens(corrupted_tokens[0]) if display_corrupted_text else model.to_str_tokens(clean_tokens[0])
  attention_patten = cache["pattern", layer_to_stare_at]
  print(f"Layer {layer_to_stare_at} Head {head_to_isolate} Activation Patterns:")

  if verbose:
    display(cv.attention.attention_heads(
      tokens=tokenized_str_tokens,
      attention= attention_patten.mean(0) if not specific else attention_patten[specific_index],
      #attention_head_names=[f"L{layer_to_stare_at}H{i}" for i in range(model.cfg.n_heads)],
    ))
  else:
    print(attention_patten.mean(0).shape)

    display(cv.attention.attention_patterns(
      tokens=tokenized_str_tokens,
      attention=attention_patten.mean(0)if not specific else attention_patten[specific_index],
      attention_head_names=[f"L{layer_to_stare_at} H{i}" for i in range(model.cfg.n_heads)],
    ))


def noising_ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    answer_tokens: Float[Tensor, "batch 2"],
) -> float:
    '''
    Given logits, returns how much the performance has been corrupted due to noising.

    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    #print(logits[-1, -1])
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt = False)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()


def denoising_ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float,
    corrupted_logit_diff: float,
    answer_tokens: Float[Tensor, "batch 2"],
) -> float:
    '''
    We calibrate this so that the value is 1 when performance got restored (i.e. same as IOI dataset),
    and 0 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits, answer_tokens, per_prompt = False)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff) + 1).item()


def path_patch_to_head(model, layer, head, clean_tokens, corrupted_tokens, noising_partial, denoising_partial, noisily = True):
  """
  Calculates Mass Results for the results of path patching form upstream heads into a downstream head's query, key, value, and pattern
  """
  head_results = []

  if noisily:
    for i in ["q", "k", "v", "pattern"]:
      model.reset_hooks() # callum library buggy
      head_results.append(path_patch(
          model,
          orig_input=clean_tokens,
          new_input=corrupted_tokens,
          sender_nodes=IterNode("z"),
          receiver_nodes=[Node(i, layer = layer, head = head)],
          patching_metric=noising_partial,
      )['z'])
  else:
    for i in ["q", "k", "v", "pattern"]:
      model.reset_hooks() # callum library buggy
      head_results.append(path_patch(
          model,
          orig_input=corrupted_tokens,
          new_input=clean_tokens,
          sender_nodes=IterNode("z"),
          receiver_nodes=[Node(i, layer = layer, head = head)],
          patching_metric=denoising_partial,
      )['z'])


  return head_results