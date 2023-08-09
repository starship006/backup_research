""""
this point of this file is to see how the name mover and backup heads act in practice (not while constrained to IOI)

"""
# %%
#%pip install git+https://github.com/neelnanda-io/TransformerLens.git
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
import tqdm.auto as tqdm
import random
from pathlib import Path

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

#%pip install plotly
#%pip install matplotlib

import plotly
import plotly.express as px
#%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
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
#%pip install git+https://github.com/neelnanda-io/neel-plotly.git
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

model_name = "gpt2-medium"
backup_storage_file_name = model_name + "_new_backup_count_storage.pickle"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    #refactor_factored_attn_matrices=True,
    device = device,
)
# %% 
torch.cuda.empty_cache()

# %% Dataest
owt_dataset = utils.get_dataset("owt")
BATCH_SIZE = 50
PROMPT_LEN = 100

all_owt_tokens = model.to_tokens(owt_dataset[0:BATCH_SIZE * 2]["text"]).to(device)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
# %%

logits, cache = model.run_with_cache(owt_tokens)

print(utils.lm_accuracy(logits, owt_tokens))
print(utils.lm_cross_entropy_loss(logits, owt_tokens))


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
    effect_directions: Union[Float[Tensor, "batch d_model"], Float[Tensor, "batch pos d_model"]],
    batch_pos_dmodel = False,
    average_across_batch = True,
    apply_ln = False, # this is broken rn idk
    use_clean_cache_for_ln = True,
    clean_cache = cache
) -> Float[Tensor, "..."]:
    '''
    Gets the avg direct effect between the correct and incorrect answer for a given
    stack of components in the residual stream. Averages across batch by default. In general,
    batch dimension should go in front of pos dimension.

    NOTE: IGNORES THE VERY LAST PREDICTION AND FIRST CLEAN TOKEN; WE DON'T KNOW THE ACTUAL PREDICTED ANSWER FOR IT!

    residual_stack: components of d_model vectors to get direct effect from
    cache: cache of activations from the model
    effect_directions: [batch, d_model] vectors in d_model space that correspond to direct effect
    batch_pos_dmodel: whether the residual stack is in the form [batch, d_model] or [batch, pos, d_model]; if so, returns pos as last dimension
    average_across_batch: whether to average across batch or not; if not, returns batch as last dimension behind pos
    '''
    batch_size = residual_stack.size(-3) if batch_pos_dmodel else residual_stack.size(-2)
    

    if apply_ln:

        cache_to_use = clean_cache if use_clean_cache_for_ln else cache

        if batch_pos_dmodel:
            scaled_residual_stack = cache_to_use.apply_ln_to_stack(residual_stack, layer=-1)
        else:
            scaled_residual_stack = cache_to_use.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    else:
        print("Not applying LN")
        scaled_residual_stack = residual_stack


    # remove the first token from the clean tokens, last token from the predictions - these will now align
    scaled_residual_stack = scaled_residual_stack[:, :, :-1, :]
    effect_directions = effect_directions[:, 1:, :]
    #print(scaled_residual_stack.shape, effect_directions.shape)

    if average_across_batch:
         # average across batch
        if batch_pos_dmodel:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch pos d_model, batch pos d_model -> ... pos"
            ) / batch_size
        else:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch d_model, batch d_model -> ..."
            ) / batch_size
    else:
        if batch_pos_dmodel:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch pos d_model, batch pos d_model -> ... batch pos"
            )
        else:
            return einops.einsum(
                scaled_residual_stack, effect_directions,
                "... batch d_model, batch d_model -> ... batch"
            ) 
            
def collect_direct_effect(cache: ActivationCache, correct_tokens: Float[Tensor, "batch seq_len"],
                           title = "Direct Effect of Heads", display = True) -> Float[Tensor, "heads batch pos"]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head on each token.
    
    returns [heads, batch, pos - 1] length tensor of direct effects of each head on each (correct) token

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
per_head_direct_effect = einops.rearrange(per_head_direct_effect, "(n_layer n_head) batch pos -> n_layer n_head batch pos", n_layer = model.cfg.n_layers, n_head = model.cfg.n_heads)


# %% Function: Look at Individual Direct Effects of Heads on various positions, batches

def dir_effects_from_sample_ablating_head(layer, head):
    """this function gets the new direct effect of all the heads when sample ablating the input head
    it uses the global cache, owt_tokens, corrupted_owt_tokens
    """

    ablate_heads = [[layer, head]]
    new_cache = act_patch(model, owt_tokens, [Node("z", layer, head) for (layer,head) in ablate_heads],
                            return_item, corrupted_owt_tokens, apply_metric_to_cache= True)

    temp = collect_direct_effect(new_cache, owt_tokens, display = False)
    ablated_per_head_batch_direct_effect = einops.rearrange(temp,
                                            "(n_layer n_heads_per_layer) batch pos -> n_layer n_heads_per_layer batch pos",
                                            n_layer = model.cfg.n_layers, n_heads_per_layer = model.cfg.n_heads)
                                            
    return ablated_per_head_batch_direct_effect


def create_scatter_of_backup_of_head(layer, head):
    """"
    this function:
    1) gets the direct effect of all the heads when sample ablating the input head
    2) gets the total accumulated backup of the head for each prompt and position
    3) plots the clean direct effect vs accumulated backup
    """
    ablated_per_head_batch_direct_effect = dir_effects_from_sample_ablating_head(layer, head)

    # 2) gets the total accumulated backup of the head for each prompt and position
    downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
    assert downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() == 0
    sum_across_all_downstream_heads = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
    
    #  3) plots the clean direct effect vs accumulated backup
    fig = go.Figure()
    scatter_plot = go.Scatter(
        x= per_head_direct_effect[layer, head].flatten().cpu(),
        y=sum_across_all_downstream_heads.flatten().cpu(),
        text=[f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN))],  # Set the hover labels to the text attribute
        mode='markers',
        marker=dict(size=2, opacity=0.8),
    )
    fig.add_trace(scatter_plot)
    fig.update_layout(
        title=f"Total Accumulated Backup of {layer}.{head} in {model_name} for each Position and Batch",
    )
    fig.update_xaxes(title = "Direct Effect of Head")
    fig.update_yaxes(title = "Total Accumulated Backup")
    fig.update_layout(width=700, height=400)
    fig.show()



create_scatter_of_backup_of_head(3,1)
# %% Function(s): Graph accumulated backup vs direct effect of head


def get_backup_per_head(topk_prompts = 0):
    """
    gets the downstream accumulated backup when ablating a head
    by default, this operates across all prompts: if topk_prompts > 0, it isolates the top_k prompts where 
    the head has the highest direct effect

    also returns the clean logit diffs (either across all prompts, or on all top_k ones)
    """
    total_accumulated_backup_per_head = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    average_clean_logit_diff = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):

            if topk_prompts > 0:
                head_direct_effects = per_head_direct_effect[layer, head]
                top_indices = topk_of_Nd_tensor(head_direct_effects, topk_prompts)
                # set average_clean_logit_diff
                sum = 0
                for batch, pos in top_indices:
                    sum += per_head_direct_effect[layer, head, batch, pos].item()
                average_clean_logit_diff[layer, head] = sum / topk_prompts
            else:
                average_clean_logit_diff[layer, head] = per_head_direct_effect[layer, head].mean((0,1)).item()

            ablated_per_head_batch_direct_effect = dir_effects_from_sample_ablating_head(layer, head)
            downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
            if topk_prompts > 0:
                backup_amount = 0.0
                for batch, pos in top_indices:
                    select_heads = downstream_change_in_logit_diff[(layer+1):, :, batch, pos]
                    # for the heads in select_heads, get the total downstream change
                    backup_amount += select_heads.sum((0,1)).item()
                backup_amount /= topk_prompts
            else:
                # use all batch and prompts
                backup_amount = downstream_change_in_logit_diff[(layer+1):].mean((0,1,2,3)).item()
            
            
            total_accumulated_backup_per_head[layer, head] = backup_amount
    
    return total_accumulated_backup_per_head, average_clean_logit_diff


def plot_accumulated_backup_per_head(top_k_to_isolate, total_accumulated_backup_per_head, direct_clean_effect_per_head):
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
    x = direct_clean_effect_per_head.flatten().cpu(),
    y = total_accumulated_backup_per_head.flatten().cpu(),
    text=[f"Layer {i[0]}, Head {i[1]}" for i in itertools.product(range(model.cfg.n_layers), range(model.cfg.n_heads))],  # Set the hover labels to the text attribute
    mode='markers',
    marker=dict(size=10, color=colors, opacity=0.8),
)
    fig.add_trace(scatter_plot)
    fig.update_layout(

        title=f"Total Accumulated Backup of Heads on top {(str(round(top_k_to_isolate / (BATCH_SIZE * (PROMPT_LEN - 1)) * 100, 2)) +'%') if top_k_to_isolate != 0 else 'All'}"
        + " Prompts vs. Average Direct Effect of Heads",
        width = 1000
    )
    fig.update_xaxes(title = "Average Direct Effect of Head")
    fig.update_yaxes(title = "Total Accumulated Backup")
    fig.show()


# %% Run Function Here:
top_k_to_isolate = 100
total_accumulated_backup_per_head, direct_clean_effect_per_head = get_backup_per_head(top_k_to_isolate)
# %% Plot the results:
plot_accumulated_backup_per_head(top_k_to_isolate, total_accumulated_backup_per_head, direct_clean_effect_per_head)
# %% Function(s): Plot backup amounts of different heads on heatmap

# ablate heads and see how much backup the downstream heads show in response!
backup_count_storage = {} 
# # load backup_count_storage from pickle file
try:
    # with open(backup_storage_file_name, 'rb') as handle:
    #     backup_count_storage = pickle.load(handle)
    backup_count_storage = {} 
except:
    backup_count_storage = {} 

model.use_attn_result = True

# %%

def calculate_head_activating_other_backup_amounts(layer, head):

    ablated_per_head_batch_direct_effect = dir_effects_from_sample_ablating_head(layer, head)
    downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
    return downstream_change_in_logit_diff

def show_backup_counts(ablate_layer, ablate_head,  tenths = 11):
    """"
    tenths = how many 0.1 increments to show in the slider
    """
   
    backup_head_counts = backup_count_storage[str(ablate_layer) + "." + str(ablate_head)]
    
    fig = go.Figure()


    # generate granular heatmap (where we have the different thresholds)
    thresholded_head_counts = torch.zeros((tenths, model.cfg.n_layers, model.cfg.n_heads))
    threshold_amounts = [i * (0.1) for i in range(0, tenths)]

    for i, threshold in enumerate(threshold_amounts):
        for layer in range(model.cfg.n_layers):
            if layer <= ablate_layer:
                thresholded_head_counts[i, layer, :] = np.NAN
            else:
                for head in range(model.cfg.n_heads):
                    thresholded_head_counts[i, layer, head] = (backup_head_counts[layer, head] > threshold).sum().item()

    # Add traces, one for each slider step
    for step in range(11):
        fig.add_trace(
            go.Heatmap(
                visible=False,
                z=thresholded_head_counts[step].cpu(),
                name="Active = " + str(step),
                colorscale="RdBu",
                zmid = 0,
                text = thresholded_head_counts[step].cpu(),
                texttemplate="%{text}",
            ),
        )


    fig.data[5].visible = True
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                #{"title": f"Percent where head had more than " + str(i / 10) + f" backup on top 10% {layer if not get_all_in_same_layer else ''}.{head if not get_all_in_same_layer else ''} Backup Instances"}
                ],  # layout attribute

            # label is i * 0.1 but formatted with only one decimal place
            label=f"{i * 0.1:.1f}"
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
    # have x-axis have each head instead of every other head labeled
    fig.update_xaxes(tickmode = "array", tickvals = list(range(model.cfg.n_heads)), ticktext = list(range(model.cfg.n_heads)))

    # same thing for y-axis and layers
    fig.update_yaxes(tickmode = "array", tickvals = list(range(model.cfg.n_layers)), 
                     ticktext = list(range(model.cfg.n_layers)), mirror = True)

    # draw a square on all the NaN values
    fig.add_shape(
        # filled Rectangle
            type="rect",
            xref="x",
            yref="y",
            x0=-0.5,
            y0=-0.5,
            x1=model.cfg.n_heads - 0.5,
            y1=ablate_layer + 0.5,
            line=dict(
                
                color="grey",
                width=0,

            ),
            fillcolor="grey",
    )

    #fig.update_yaxes(mirror = True)


    #change title
    fig.update_layout(title_text=f"Backup Head Counts by Prompt and Threshold of {ablate_layer}.{ablate_head}") 
    fig.show()

# %% Store data in backup_count_storage
for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for head in tqdm.tqdm(range(model.cfg.n_heads)):
        if str(layer) + "." + str(head) in backup_count_storage.keys():
            print("Lol")
        else:
            backup_count_storage[str(layer) + "." + str(head)] = calculate_head_activating_other_backup_amounts(layer, head)
# %% See Heatmap
show_backup_counts(7,2) # TODO: error where if the loaded dictionary is not the same prompt_len or batch_size as what the file is, it can lead to errors

# %% Close out work


model.use_attn_result = False
with open(backup_storage_file_name, 'wb') as handle:
    print("DONE")
    pickle.dump(backup_count_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %% Function to make graphs
# %%
import networkx as nx
import matplotlib.pyplot as plt

# %%


def draw_backup_graph(backup_threshold = 0.25, frequency_threshold = 50):
    G = nx.DiGraph()
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            source_nodes = [layer, head]
            destination_nodes = [[i,j] for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads) if 
                             ((backup_count_storage[f"{layer}.{head}"][i][j]) > backup_threshold).sum().item() > frequency_threshold]
            if [layer, head] in destination_nodes:
                destination_nodes.remove([layer, head])
            for dest in destination_nodes:
                G.add_edge(tuple(source_nodes), tuple(dest))

    # # add summed version
    # backup_head_counts = torch.zeros((11, model.cfg.n_layers, model.cfg.n_heads))
    # for i in range(model.cfg.n_heads):
    #     backup_head_counts += backup_count_storage[str(layer) + "." + str(i)]
    # destination_nodes = [[i,j] for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads) if backup_head_counts[2][i][j] > 20]
    # for dest in destination_nodes:
    #     G.add_edge("Layer " + str(layer), tuple(dest))


    # define positions according to layer
    pos = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            pos[(layer, head)] = (head * 2 + 10, layer)

        pos["Layer " + str(layer)] = (model.cfg.n_heads * 2 + 10, layer)
    # make graph look nice


    nx.draw_networkx(G, pos, with_labels=True, node_size=200, node_color='skyblue', edge_color='gray',
        arrowsize=20, font_size=5, font_weight='bold', font_color='black')
    plt.show()


# %% Run backup graph:
draw_backup_graph(0.3, 30)
# %% See what percentage of backup behaviour when ablating a head is explained by downstream heads

def backup_explained_by_head(ablate_layer, ablate_head, top_k = 100, heads_show = 2):
    """in the top_k activations of the ablated head, see how much specific heads can explain the downstream backup behaviour"""
    
    # get top_k activations of ablated head
    head_direct_effects = per_head_direct_effect[ablate_layer, ablate_head]
    top_indices = topk_of_Nd_tensor(head_direct_effects, top_k)
    backup_amounts = backup_count_storage[str(ablate_layer) + "." + str(ablate_head)]
    
    thresholds = [i * 0.01 for i in range(101)]
    num_thresholds = len(thresholds)

    explained_by_head = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, num_thresholds))

    for (batch, pos) in top_indices:
        total_backup_in_pos = backup_amounts[..., batch, pos].sum().item() - backup_amounts[ablate_layer, ablate_head, batch, pos].item()
        for layer in range(ablate_layer, model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                for threshold in range(num_thresholds): # make sure this is analogous to below summed
                    if backup_amounts[layer, head, batch, pos].item() / total_backup_in_pos >= thresholds[threshold]:
                        explained_by_head[layer, head, threshold] += 1
                

    # get highest activating head
    top_heads = topk_of_Nd_tensor(explained_by_head.sum(-1), heads_show)

    # remove own head if it is included
    if [ablate_layer, ablate_head] in top_heads:
        top_heads = topk_of_Nd_tensor(explained_by_head.sum(-1), heads_show + 1)
        top_heads.remove([ablate_layer, ablate_head])


    fig = go.Figure()

    for head in top_heads:
    
        activations = explained_by_head[head[0], head[1]]
    
        fig.add_trace(
            go.Scatter(
                x = thresholds,
                y = activations.cpu() / top_k,
                name = f"{head[0]}.{head[1]}"
            )
        )

    # summed
    total_backup = torch.zeros(num_thresholds)
    for (batch, pos) in top_indices:
        total_backup_in_pos = backup_amounts[..., batch, pos].sum().item() - backup_amounts[ablate_layer, ablate_head, batch, pos].item()
        
        for threshold in range(num_thresholds): # make sure this is analogous to above individual
            sum = 0
            for layer, head in top_heads:
                sum += backup_amounts[layer, head, batch, pos].item()
            if sum / total_backup_in_pos >= thresholds[threshold]:
                total_backup[threshold] += 1


    fig.add_trace(
        go.Scatter(
            
            x = thresholds,
            y = total_backup.cpu() / top_k,
            name = "Summed",
        )
    )
    fig.update_layout(
        title=f"Percent Backup Explained by Heads for {ablate_layer}.{ablate_head} in {model_name} for top {top_k} activations",
    )
    fig.update_xaxes(title = "Percent of Total Backup Explained")
    fig.update_yaxes(title = "Percent Samples Explained",
                     #set range to 0 to 1
                     range = [0, 1])
    # have percentages on y axis
    fig.update_yaxes(tickformat = "%")

    fig.show()

# %%


backup_explained_by_head(9,6, 100, 3)
# %% Isolating different prompts which heavily activate mover heads!


# %%
def view_top_prompts_by_head(layer, head, topk = 10):
    head_direct_effects = per_head_direct_effect[layer, head]
    top_indicies = topk_of_Nd_tensor(head_direct_effects, topk)
    for batch, pos_minus_one_location in top_indicies:
        prompt_up_till_word = model.to_string(owt_tokens[batch][:pos_minus_one_location + 1])
        predicted_word = model.to_string(owt_tokens[batch, pos_minus_one_location + 1])
        print("--------------------------  " + str(batch) + "\n-------------------------- \n Logit Contribution: " + str(head_direct_effects[batch, pos_minus_one_location].item()))
        print(prompt_up_till_word + "  ---------> " + predicted_word)
        pass
    print(top_indicies)


def view_prompts_by_head_batch_pos(list_of_locations, backup_amounts):
    index = 0
    for layer, head, batch, pos in list_of_locations:
        prompt_up_till_word = model.to_string(owt_tokens[batch][:pos + 1])
        predicted_word = model.to_string(owt_tokens[batch, pos + 1])

        print()
        print()
        print("----------------------------------------------------------------------------")
        print(f"Layer {layer}, Head {head}, Batch {batch}, Pos {pos}")
        print("\n Logit Contribution: " + str(per_head_direct_effect[layer, head, batch, pos].item()) + "\n Backup Amount: " + str(backup_amounts[index]))
        print(prompt_up_till_word + "  ---------> " + predicted_word)
        index += 1
        
# %%
view_top_prompts_by_head(9,6, topk = 40)
# %%
batch = 45

plotly.io.renderers.default = "vscode"
cv.activations.text_neuron_activations(
    model.to_str_tokens(owt_tokens[batch, 1:]),
    einops.rearrange(per_head_direct_effect[:, :, batch, :], "layer head pos -> pos layer head"),
    second_dimension_name = "Head"
)
# %% 
def get_responses_to_ablation_of_heads() -> Float[Tensor, "ablated_layer ablated_head downstream_layer downstream_head batch pos"]:
    results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN-1))
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            ablated_per_head_batch_direct_effect = dir_effects_from_sample_ablating_head(layer, head)
            downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
            results[layer, head] = downstream_change_in_logit_diff
    return results


all_ablation_results = get_responses_to_ablation_of_heads()

# %%
def get_prompts_where_head_is_most_backup(layer, head, all_ablation_results = all_ablation_results, topk=10):
    """
    look across all batches and positions and find when the head is doing the most backup
    """

    head_backup_amounts = all_ablation_results[:, :, layer, head, :, :]
    top_indices = topk_of_Nd_tensor(head_backup_amounts, topk)
    #print(top_indices)
    return top_indices
    


# %%
target_layer, target_head = 11, 7
indices_of_highest_backup = get_prompts_where_head_is_most_backup(target_layer,target_head, topk = 200)
backup_amounts = [all_ablation_results[:, :, target_layer, target_head, :, :][a,b,c,d].item() for (a,b,c,d) in indices_of_highest_backup]
view_prompts_by_head_batch_pos(indices_of_highest_backup, backup_amounts)
# %%
temp = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for i in indices_of_highest_backup:
    temp[i[0], i[1]] += 1
imshow(temp, title = f"what heads is {target_layer}.{target_head} backing up?", width = 500, labels = {"x": "Head", "y": "Layer", "color": "Frequency"})
# %%
def get_non_top_prompts(send_layer, send_head, heads_to_ignore, topk=10, all_ablation_results = all_ablation_results):
    index = 0
    head_backup_amounts = all_ablation_results[send_layer, send_head, :, : :, :].clone()
    for head in heads_to_ignore:
        print(head)
        #print(head_backup_amounts[head[0], head[1]].shape)
        #print(torch.zeros((BATCH_SIZE, PROMPT_LEN - 1)).shape)
        head_backup_amounts[head[0], head[1]] = torch.zeros((BATCH_SIZE, PROMPT_LEN - 1))
    top_indices = topk_of_Nd_tensor(head_backup_amounts, topk)
    print(top_indices)
    print(head_backup_amounts[10,2])
    for layer, head, batch, pos in top_indices:
        prompt_up_till_word = model.to_string(owt_tokens[batch][:pos + 1])
        predicted_word = model.to_string(owt_tokens[batch, pos + 1])
        backup_amount = head_backup_amounts[layer, head, batch, pos].item()

        print()
        print()
        print("----------------------------------------------------------------------------")
        print(f"Layer {layer}, Head {head}, Batch {batch}, Pos {pos}")
        print("\n Logit Contribution: " + str(per_head_direct_effect[layer, head, batch, pos].item()) + "\n Backup Amount: " + str(backup_amount))
        print(prompt_up_till_word + "  ---------> " + predicted_word)
        index += 1


# %%
layer = 9
head = 6
heads_to_spare = [[10,2]]
ignore_heads = [(i,j) for i in range(layer, model.cfg.n_layers) for j in range(model.cfg.n_heads) if [i,j] not in heads_to_spare]
get_non_top_prompts(layer, head, ignore_heads, topk = 4)
# %%

# %%
# 
# 

def get_slope_of_best_fit_line(layer, head, graph = True):
    """"
    this function:
    1) idea would be to filter for points that are greater than 0.5 direct effect, and then make a line of best fit
    2) plot line and points
    """

    ablated_per_head_batch_direct_effect = dir_effects_from_sample_ablating_head(layer, head)

    # gets the total accumulated backup of the head for each prompt and position
    downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
    assert downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() == 0 # layer 'layer' will not be 0
    sum_across_all_downstream_heads = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
    
    extreme_dir_effects = []
    parallel_backup_amounts = []

    for x, y in zip(per_head_direct_effect[layer, head].flatten().cpu(), sum_across_all_downstream_heads.flatten().cpu()):
        
        extreme_dir_effects.append(x)
        parallel_backup_amounts.append(y)

    # get slope of line of best fit with numpy
    if len(extreme_dir_effects) == 0:
        slope = 0
        intercept = 0
    else:
        # just get a best fit line
        # slope, intercept = np.polyfit(extreme_dir_effects, parallel_backup_amounts, 1)

        # get a best fit line with a constraint that it must go through the origin
        slope, intercept = np.linalg.lstsq(np.vstack([extreme_dir_effects, np.ones(len(extreme_dir_effects))]).T, parallel_backup_amounts, rcond=None)[0]

    if graph:
        fig = go.Figure()
        scatter_plot = go.Scatter(
            x= per_head_direct_effect[layer, head].flatten().cpu(),
            y=sum_across_all_downstream_heads.flatten().cpu(),
            text=[f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN))],  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
        )
        fig.add_trace(scatter_plot)

        # get maximum direct effect
        max_x = max(per_head_direct_effect[layer, head].flatten().cpu())

        # add line of best fit
        fig.add_trace(go.Scatter(
            x=torch.linspace(0,max_x,100),
            y=torch.linspace(0,max_x,100) * slope + intercept,
            mode='lines',
            name='lines'
        ))
        fig.update_layout(
            title=f"Total Accumulated Backup of {layer}.{head} in {model_name} for each Position and Batch",
        )
        fig.update_xaxes(title = "Direct Effect of Head")
        fig.update_yaxes(title = "Total Accumulated Backup")
        fig.update_layout(width=700, height=400)
        fig.show()


    return slope, intercept
# %%
get_slope_of_best_fit_line(6,4)
# %%
slopes_of_head_backup = torch.zeros((12,12))
for layer in tqdm.tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        slopes_of_head_backup[layer, head] = get_slope_of_best_fit_line(layer, head, graph = False)[0]
# %%

imshow(slopes_of_head_backup, title = "Slopes of Head Backup",
       text_auto = True, width = 800, height = 800)# show a number above each square)

# %%

# do a scatter of average direct effect vs slope
top_k_to_isolate = 100
# direct_clean_effect_per_head use this from above

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
    x= mean_per_head_direct_effect.flatten().cpu(),
    y=slopes_of_head_backup.flatten().cpu(),
    text=[f"{i}.{j}" for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)],
    marker=dict(size=10, color=colors, opacity=0.8,),
    mode='markers',
)

fig.add_trace(scatter_plot)
fig.update_layout(
    title=f"Average Direct Effect of Heads on top {str(round(top_k_to_isolate / (BATCH_SIZE * (PROMPT_LEN - 1)) * 100, 2)) +'%'} vs. Slope of Best Fit Line of Backup",
)

fig.update_xaxes(title = "Average Direct Effect of Head")
fig.update_yaxes(title = "Slope of Best Fit Line of Backup")
fig.update_layout(width=700, height=400)
fig.show()
# %%
