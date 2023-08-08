
import pickle


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



from path_patching import Node, IterNode, path_patch, act_patch

# %%
from neel_plotly import imshow, line, scatter, histogram


# %%
import tqdm
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

def residual_stack_to_direct_effect(
    residual_stack: Union[Float[Tensor, "... batch d_model"], Float[Tensor, "... batch pos d_model"]],
    cache: ActivationCache,
    effect_directions: Float[Tensor, "batch d_model"],
    batch_pos_dmodel = False,
    average_across_batch = True,
    apply_ln = False # this is broken rn idk
) -> Float[Tensor, "..."]:
    '''
    Gets the avg direct effect between the correct and incorrect answer for a given
    stack of components in the residual stream. Averages across batch by default. In general,
    batch dimension should go in front of pos dimension.

    residual_stack: components of d_model vectors to get direct effect from
    cache: cache of activations from the model
    effect_directions: [batch, d_model] vectors in d_model space that correspond to direct effect
    batch_pos_dmodel: whether the residual stack is in the form [batch, d_model] or [batch, pos, d_model]; if so, returns pos as last dimension
    average_across_batch: whether to average across batch or not; if not, returns batch as last dimension behind pos
    '''
    batch_size = residual_stack.size(-3) if batch_pos_dmodel else residual_stack.size(-2)
    

    if apply_ln:
        if batch_pos_dmodel:
            #print(cache["ln_final.hook_scale"].shape)
            #print(residual_stack.shape)
            #scaled_residual_stack = torch.zeros(residual_stack.shape).cuda()
            # for pos in range(residual_stack.shape[2]):
            #     scaled_residual_stack[:, :, pos, :] = cache.apply_ln_to_stack(residual_stack[:, :, pos, :], layer=-1, pos_slice = -1)
            scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1)
            #print('passed')
        else:
            scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    else:
        print("Not applying LN")
        scaled_residual_stack = residual_stack

    if not average_across_batch:
        if not batch_pos_dmodel:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch d_model, batch d_model -> ... batch"
            ) 
        else:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch pos d_model, batch pos d_model -> ... batch pos"
            ) 
    else:
        # average across batch
        if not batch_pos_dmodel:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch d_model, batch d_model -> ..."
            ) / batch_size
        else:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch pos d_model, batch pos d_model -> ... pos"
            ) / batch_size


def collect_direct_effect(cache: ActivationCache, model, correct_tokens: Float[Tensor, "batch seq_len"],
                           title = "Direct Effect of Heads", display = True) -> Float[Tensor, "heads batch pos"]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head on each token.

    cache: cache of activations from the model
    correct_tokens: [batch, seq_len] tensor of correct tokens
    title: title of the plot (relavant if display == True)
    display: whether to display the plot or return the data; if False, returns [head, pos] tensor of direct effects
    """

    
    clean_per_head_residual: Float[Tensor, "head batch seq d_model"] = cache.stack_head_results(layer = -1, return_labels = False, apply_ln = False) 
    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(correct_tokens)

    # get the direct effect of heads by positions
    per_head_direct_effect: Float[Tensor, "heads batch pos"] = residual_stack_to_direct_effect(clean_per_head_residual,
                                                                                          cache, token_residual_directions,
                                                                                          batch_pos_dmodel = True, average_across_batch = False,
                                                                                          apply_ln = True)
   
    #assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1])

    if display:    
        mean_per_head_direct_effect = per_head_direct_effect.mean(dim = (1,2))
        mean_per_head_direct_effect = einops.rearrange(mean_per_head_direct_effect, "(n_layer n_heads_per_layer) -> n_layer n_heads_per_layer",
                                                   n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
        fig = imshow(
            torch.stack([mean_per_head_direct_effect]),
            return_fig = True,
            facet_col = 0,
            facet_labels = [f"Direct Effect of Heads"],
            title=title,
            labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
            #coloraxis=dict(colorbar_ticksuffix = "%"),
            border=True,
            width=500,
            margin={"r": 100, "l": 100}
        )
        return per_head_direct_effect     
    else:
        return per_head_direct_effect
    
def return_item(item):
  return item