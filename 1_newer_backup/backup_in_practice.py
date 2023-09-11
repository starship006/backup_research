# %%
#%pip install git+https://github.com/neelnanda-io/TransformerLens.git
#%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
# %%
#%pip install plotly
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



# %%
#!sudo apt install unzip
if not os.path.exists("path_patching.py"):
        !wget https://github.com/callummcdougall/path_patching/archive/refs/heads/main.zip
        !unzip main.zip 'path_patching-main/ioi_dataset.py'
        !unzip main.zip 'path_patching-main/path_patching.py'
        sys.path.append("path_patching-main")
        os.remove("main.zip")
        os.rename("path_patching-main/ioi_dataset.py", "ioi_dataset.py")
        os.rename("path_patching-main/path_patching.py", "path_patching.py")
        os.rmdir("path_patching-main")

from path_patching import Node, IterNode, path_patch, act_patch

# %%
%import neel_plotly
from neel_plotly import imshow, line, scatter, histogram


# %%
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
    if "color_continuous_midpoint" not in kwargs_pre:
        fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, **kwargs_pre)
    else:
        fig = px.imshow(utils.to_numpy(tensor), **kwargs_pre)
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

# def hist(tensor, renderer=None, **kwargs):
#     kwargs_post = {k: v for k, v in kwargs.items() if k in update_layout_set}
#     kwargs_pre = {k: v for k, v in kwargs.items() if k not in update_layout_set}
#     names = kwargs_pre.pop("names", None)
#     if "barmode" not in kwargs_post:
#         kwargs_post["barmode"] = "overlay"
#     if "bargap" not in kwargs_post:
#         kwargs_post["bargap"] = 0.0
#     if "margin" in kwargs_post and isinstance(kwargs_post["margin"], int):
#         kwargs_post["margin"] = dict.fromkeys(list("tblr"), kwargs_post["margin"])
#     fig = px.histogram(x=tensor, **kwargs_pre).update_layout(**kwargs_post)
#     if names is not None:
#         for i in range(len(fig.data)):
#             fig.data[i]["name"] = names[i // 2]
#     fig.show(renderer)

# %%
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# %%
model_name = "gpt2-medium"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,

    device = device
)

file_path = f"backup_count_storage_{model_name}.pickle"


# %%
owt_dataset = utils.get_dataset("pile")

# %%
owt_dataset
# %%
BATCH_SIZE = 150
PROMPT_LEN = 20

all_owt_tokens = model.to_tokens(owt_dataset[0:BATCH_SIZE * 2]["text"])
print(all_owt_tokens.shape)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]

owt_tokens.shape
assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)


# %%
logits, cache = model.run_with_cache(owt_tokens)

print(utils.lm_accuracy(logits, owt_tokens))
print(utils.lm_cross_entropy_loss(logits, owt_tokens))
# %%
utils.test_prompt(owt_dataset[0]["text"][0:30], " will", model)


# %% [markdown]
# # Helper Functions
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


def collect_direct_effect(cache: ActivationCache, correct_tokens: Float[Tensor, "batch seq_len"],
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

per_head_direct_effect = collect_direct_effect(cache, owt_tokens, display = True)


# %%
per_head_batch_direct_effect = einops.rearrange(per_head_direct_effect,
                                         "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                           n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)


per_head_direct_effect = einops.rearrange(per_head_direct_effect,
                                         "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer (batch pos)",
                                           n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)

# %%
# plot a histogram of the values of the direct effect at [11, 8] 
fig = go.Figure()

# set number of bins, and make graphs overlap
fig.update_layout(barmode='overlay', )

layer_to_plot = model.cfg.n_layers - 1
for i in range(model.cfg.n_heads):
     fig.add_trace(
          go.Histogram(
               x=per_head_direct_effect[layer_to_plot, i, :].cpu(),
                    name=f"({layer_to_plot}.{i})",
                    opacity = 0.6,
                    nbinsx= int(100 * (per_head_direct_effect[-1, i, :].max().item() 
                                       - per_head_direct_effect[-1, i, :].min().item())),
          ),
          #row=1, col=1
     )

# add marginal


fig.update_layout(
     title="Direct Effect of Heads",
     xaxis_title="Logit Contribution",
     yaxis_title="Count",
     width=1000,
     height=600,
     
     #coloraxis=dict(colorbar_ticksuffix = "%"),
     #facet_col=0,
     #facet_labels=["Direct Effect of Heads"],
     #border=True,
)
# update x-axis range
fig.update_xaxes(range=[-1, 1])
fig.show()

# %%
# # checking diffs on a per-prompt basis
#example_prompt = topk_of_Nd_tensor(per_head_batch_direct_effect[4,5].mean(-1),20)[19][0]

mean_clean_results = per_head_batch_direct_effect[..., 0:20, :].mean((-2,-1))

# get top heads
top_heads = topk_of_Nd_tensor(mean_clean_results, 3)
ablate_heads = [[0,2], [0,5]]#[i for i in top_heads if i[0] != 11]
new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in ablate_heads], return_item, corrupted_owt_tokens, apply_metric_to_cache= True)


temp = collect_direct_effect(new_cache, owt_tokens, display = False)
ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                        "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                        n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
ablated_per_head_direct_effect = einops.rearrange(temp,
                                        "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer (batch pos)",
                                        n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)


mean_ablate_results = ablated_per_head_batch_direct_effect[..., 0:20, :].mean((-2,-1))

imshow(torch.stack([mean_ablate_results - mean_clean_results, mean_clean_results]), facet_col = 0, title = "Difference in Direct Effect of Heads, while also ablating " + str(ablate_heads), border = True, width = 1000, height = 600)
#imshow(mean_clean_results, title = "Difference in Direct Effect of Heads", border = True, width = 1000, height = 600)


# %%

# AVERAGE ACROSS THE BIG TIME PROMPTS
for i in range(model.cfg.n_heads):
    ablate_heads = [[1,i]]
    prompts_to_use = [i[0] for i in topk_of_Nd_tensor(per_head_batch_direct_effect[ablate_heads[0][0],ablate_heads[0][1]].mean(-1),20)[:]]
    mean_clean_results = per_head_batch_direct_effect[..., prompts_to_use, :].mean(-1)



    # get top heads
    new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in ablate_heads], return_item, corrupted_owt_tokens, apply_metric_to_cache= True)


    temp = collect_direct_effect(new_cache, owt_tokens, display = False)
    ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                            "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                            n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
    ablated_per_head_direct_effect = einops.rearrange(temp,
                                            "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer (batch pos)",
                                            n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)


    mean_ablate_results = ablated_per_head_batch_direct_effect[..., prompts_to_use, :].mean(-1)


    # now average across the top prompts
    mean_clean_results = mean_clean_results.mean(-1)
    mean_ablate_results = mean_ablate_results.mean(-1)

    #imshow(torch.stack([mean_ablate_results - mean_clean_results, mean_clean_results]), facet_col = 0, title = "Difference in Direct Effect of Heads, while also ablating " + str(ablate_heads), border = True, width = 1000, height = 600)
    
    # print indices of mean_ablate_results - mean_clean_results for which the value is great than 0.15
    top_backup = topk_of_Nd_tensor(mean_ablate_results - mean_clean_results, 10)
    
    
    print([i for i in top_backup if (mean_ablate_results[i[0], i[1]] - mean_clean_results[i[0], i[1]]) > 0.02])

    print(str(i) + "^^^^^^^^")
# %%

# is there a general pattern to which you can expect backup? or when a head activates most?

# controlling for head biases
mean_direct_effect_per_heads = per_head_batch_direct_effect.mean((-2, -1))
#print(einops.repeat(mean_direct_effect_per_heads, "layer head -> layer head i", i = 3000).shape)
#print(per_head_batch_direct_effect.shape)

per_head_diff_activations: Float[Tensor, "layer head batch pos"] = per_head_batch_direct_effect - einops.repeat(mean_direct_effect_per_heads, "layer head -> layer head i j", i = 1, j = 1)




# %%

def visualize_backup_from_top_act_diff(layer, head):
    # currently doesn't look at pos individually
    ablate_head = [layer, head]
    head_mean_diffs = per_head_diff_activations[ablate_head[0], ablate_head[1]].mean(-1)
    top_prompts_indices: Float[Tensor, "batch"] = topk_of_Nd_tensor(head_mean_diffs, 10)
    
    

    new_cache = act_patch(model, owt_tokens, [Node("z", ablate_head[0], ablate_head[1])], return_item, corrupted_owt_tokens, apply_metric_to_cache= True)


    temp = collect_direct_effect(new_cache, owt_tokens, display = False)
    ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                            "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                            n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
 


    print(per_head_batch_direct_effect.shape)
    mean_clean_results = per_head_batch_direct_effect[..., top_prompts_indices, :]
    mean_ablate_results = ablated_per_head_batch_direct_effect[..., top_prompts_indices, :]

    # now average across the top prompts

    print(mean_clean_results.shape)
    print(mean_ablate_results.shape)

    mean_clean_results = mean_clean_results.mean((-3,-2,-1))
    mean_ablate_results = mean_ablate_results.mean((-3, -2,-1)) # TODO weird bug


    print(mean_clean_results.shape)
    print(mean_ablate_results.shape)

    imshow(torch.stack([mean_ablate_results - mean_clean_results, mean_clean_results]), facet_col = 0, title = "Difference in Direct Effect of Heads, while also ablating " + str(ablate_head), border = True, width = 1000, height = 600)
    
    # print indices of mean_ablate_results - mean_clean_results for which the value is great than 0.15
    top_backup = topk_of_Nd_tensor(mean_ablate_results - mean_clean_results, 10)
    
    
    print([i for i in top_backup if (mean_ablate_results[i[0], i[1]] - mean_clean_results[i[0], i[1]]) > 0.02])

# %%

# part 1: plot mean avg output of head against average total change in direct effect of head upon ablation

total_accumulated_backup_per_head = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for i in range(model.cfg.n_layers):
    for j in range(model.cfg.n_heads):
        #total_accumulated_backup_per_head[i,j] = per_head_batch_direct_effect[i,j].mean((-2,-1)).sum()

        ablate_heads = [[i,j]]
        prompts_to_use = [i[0] for i in topk_of_Nd_tensor(per_head_batch_direct_effect[ablate_heads[0][0],ablate_heads[0][1]].mean(-1),20)[:]]
        mean_clean_results = per_head_batch_direct_effect[..., prompts_to_use, :].mean(-1)


        # get top heads
        new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in ablate_heads], return_item, corrupted_owt_tokens, apply_metric_to_cache= True)


        temp = collect_direct_effect(new_cache, owt_tokens, display = False)
        ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                                "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                                n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
        ablated_per_head_direct_effect = einops.rearrange(temp,
                                                "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer (batch pos)",
                                                n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)


        mean_ablate_results = ablated_per_head_batch_direct_effect[..., prompts_to_use, :].mean(-1)


        # now average across the top prompts
        mean_clean_results = mean_clean_results.mean(-1)
        mean_ablate_results = mean_ablate_results.mean(-1)
        backup_results = mean_ablate_results - mean_clean_results

        total_accumulated_backup_per_head[i,j] = backup_results.sum() - backup_results[i,j]

# %%

# graph per_head_batch_direct_effect by total_accumulated_backup_per_head
print(mean_direct_effect_per_heads[7,8])
fig = go.Figure()

colors = ['rgb(0, 0, 0)'] * model.cfg.n_layers * model.cfg.n_heads  # Initialize with black color
group_size = model.cfg.n_layers
num_groups = model.cfg.n_layers * model.cfg.n_heads // group_size
color_step = 255 / num_groups
for i in range(num_groups):
    start_index = i * group_size
    end_index = (i + 1) * group_size
    r = int(i * color_step)
    g = int(i * color_step)
    b = int(i * color_step / 10)
    color = f'rgb({r}, {g}, {b})'
    colors[start_index:end_index] = [color] * group_size




scatter_plot = go.Scatter(
    x= mean_direct_effect_per_heads.flatten().cpu(),
    y=total_accumulated_backup_per_head.flatten().cpu(),
    text=[f"Layer {i[0]}, Head {i[1]}" for i in itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads))],  # Set the hover labels to the text attribute
    mode='markers',
    marker=dict(size=10, color=colors, opacity=0.8),
    
)
fig.add_trace(scatter_plot)
fig.update_layout(
    title="Total Accumulated Backup of Heads on Top-20 Prompts vs. Average Direct Effect of Heads",
)
fig.update_xaxes(title = "Average Direct Effect of Head")
fig.update_yaxes(title = "Total Accumulated Backup")
fig.show()
# %%
#backup_count_storage = {}   # for storing the results of head ablations and however much other stuff turns on

# # load backup_count_storage from pickle file
# with open('backup_count_storage.pickle', 'rb') as handle:
#     backup_count_storage = pickle.load(handle)

# %%

model.use_attn_result = True

def calculate_head_activating_other_backup_amounts(layer, head, num_breakdown = 10):

    three_one_direct_effects = per_head_batch_direct_effect[layer, head]
    aggregate_backup = torch.zeros(three_one_direct_effects.shape)

    # gather aggregate_backup of the head
    for pos_index in range(three_one_direct_effects.shape[-1]):
        new_cache = act_patch(model, owt_tokens, [Node("z", layer, head, seq_pos=pos_index)], return_item, corrupted_owt_tokens, apply_metric_to_cache= True)

        temp = collect_direct_effect(new_cache, owt_tokens, display = False)
        ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                                "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                                n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
        ablated_per_head_direct_effect = einops.rearrange(temp,
                                                "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer (batch pos)",
                                                n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)

        
        backup_effect = ablated_per_head_batch_direct_effect - per_head_batch_direct_effect
        three_one_summed_ablations = backup_effect.sum((0, 1))


        aggregate_backup[:, pos_index] = three_one_summed_ablations[:, pos_index] - backup_effect[layer, head,:,pos_index]

    #fig = go.Figure()
    # scatter_plot = go.Scatter(
    #     x= three_one_direct_effects.flatten().cpu(),
    #     y=aggregate_backup.flatten().cpu(),
    #     text=[f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(150), range(20))],  # Set the hover labels to the text attribute
    #     mode='markers',
    #     marker=dict(size=2, opacity=0.8),
    # )
    # fig.add_trace(scatter_plot)
    # fig.update_layout(
    #     title=f"Total Accumulated Backup of {layer}.{head} by Position and Batch when sample ablating in position at {layer}.{head}",
    # )
    # fig.update_xaxes(title = "Direct Effect of Head")
    # fig.update_yaxes(title = "Total Accumulated Backup")
    # fig.show()
   

    # get highest backup activating prompts
    total_top_examples = int(aggregate_backup.flatten().shape[0] / 30)
    top_backup_locations = topk_of_Nd_tensor(aggregate_backup, total_top_examples)

    increment = 1 / num_breakdown
  
    model.set_use_attn_result(True)
    backup_head_counts = torch.zeros((num_breakdown, model.cfg.n_layers, model.cfg.n_heads))

    # for each top backup prompt and location, see what heads are activated (past different thresholds)
    for batch, pos in top_backup_locations:
        
        #print(model.to_str_tokens(owt_tokens[batch, 0:pos+2]))
        new_cache = act_patch(model, owt_tokens, [Node("z", layer, head, seq_pos=pos)], return_item, corrupted_owt_tokens, apply_metric_to_cache= True)

        temp = collect_direct_effect(new_cache, owt_tokens, display = False)
        ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                                "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                                n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
        specific_example = ablated_per_head_batch_direct_effect[:, : , batch, pos]
        clean_version = per_head_batch_direct_effect[:, :, batch, pos]
        
        
        for tenth in range(num_breakdown):
            backup_heads_prompt = [[i,j] for i in range(clean_version.shape[0]) 
                for j in range(clean_version.shape[1]) if (specific_example - clean_version)[i,j] > (tenth * increment)]
            
            for i,j in backup_heads_prompt:
                backup_head_counts[tenth, i,j] += 1
                

    backup_count_storage[str(layer) + "." + str(head)] = backup_head_counts
    return backup_head_counts

#%%


# see if file_patch exists

try:
    if os.path.getsize(file_path) > 0:
        with open(file_path, 'rb') as my_file:
            unpickler = pickle.Unpickler(my_file)
            backup_count_storage = unpickler.load()
            print(backup_count_storage)
    else:
        print('The file is empty')
        backup_count_storage = {}
except:
    backup_count_storage = {}
#%%
    
for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for head in tqdm.tqdm(range(model.cfg.n_heads)):
        if str(layer) + "." + str(head) in backup_count_storage.keys():
            print("Lol")
        else:
            backup_count_storage[str(layer) + "." + str(head)] = calculate_head_activating_other_backup_amounts(layer, head)

# %%
# # download backup_count_storage
# with open('backup_count_storage.pickle', 'wb') as handle:
#     pickle.dump(backup_count_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

# use pickle to download current backup_count_storage_into the file
with open(file_path, 'wb') as handle:
    print("DONE")
    pickle.dump(backup_count_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%

def show_backup_counts(layer, head, get_all_in_same_layer = False):


    if get_all_in_same_layer:
        backup_head_counts = torch.zeros((11, model.cfg.n_layers, model.cfg.n_heads))
        for i in range(model.cfg.n_heads):
            backup_head_counts += backup_count_storage[str(layer) + "." + str(i)]

    else:
        backup_head_counts = backup_count_storage[str(layer) + "." + str(head)]
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in range(11):
        fig.add_trace(
            go.Heatmap(
                visible=False,
                z=backup_head_counts[step].cpu(),
                name="Active = " + str(step),
                colorscale="RdBu",
                zmid = 0,
                text = backup_head_counts[step].cpu(),
                texttemplate="%{text}",
            ),
        )


    fig.data[5].visible = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": f"Percent where head had more than " + str(i / 10) + f" backup on top 10% {layer if not get_all_in_same_layer else ''}.{head if not get_all_in_same_layer else ''} Backup Instances"}],  # layout attribute
            label = str(i / 10)
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=5,
        currentvalue={"prefix": "Threshold: ",},
        pad={"t": 50},
        steps=steps,
    )]

    fig.update_layout(
        sliders=sliders,
        width = 800
    )

    #change title
    fig.update_layout(title_text=f"Backup Head Counts by Prompt and Threshold of {layer}.{head}") 


    fig.show()
# %%
# using pickle, save backup_head_counts


# %%

for i in range(model.cfg.n_heads):
    show_backup_counts(5, i, True)

# %%

# understanding what 7.8 is doing!

backup_count_storage.keys()




# %%
show_backup_counts(0,8, True)
# %%
import networkx as nx
import matplotlib.pyplot as plt

# %%
# create a directed graph from this
G = nx.DiGraph()


for layer in range(7):
    for head in range(model.cfg.n_heads):
        source_nodes = [layer, head]
        destination_nodes = [[i,j] for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads) if backup_count_storage[f"{layer}.{head}"][2][i][j] > 20]
        
        for dest in destination_nodes:
            G.add_edge(tuple(source_nodes), tuple(dest))

    # add summed version
    backup_head_counts = torch.zeros((11, model.cfg.n_layers, model.cfg.n_heads))
    for i in range(model.cfg.n_heads):
        backup_head_counts += backup_count_storage[str(layer) + "." + str(i)]
    destination_nodes = [[i,j] for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads) if backup_head_counts[2][i][j] > 20]
    for dest in destination_nodes:
        G.add_edge("Layer " + str(layer), tuple(dest))

# %%
# define positions according to layer
pos = {}
for layer in range(8):
    for head in range(model.cfg.n_heads):
        pos[(layer, head)] = (head * 2 + 10, layer)

    pos["Layer " + str(layer)] = (model.cfg.n_heads * 2 + 10, layer)
# make graph look nice


nx.draw_networkx(G, pos, with_labels=True, node_size=200, node_color='skyblue', edge_color='gray',
        arrowsize=20, font_size=5, font_weight='bold', font_color='black')
plt.show()

# %%


backup_count_storage[f"{3}.{1}"][5].shape
# %%
backup_count_storage.keys()
# %%



# %%
