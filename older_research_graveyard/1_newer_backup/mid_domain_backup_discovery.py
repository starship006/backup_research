# %%


# %%
%pip install git+https://github.com/neelnanda-io/TransformerLens.git
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

%pip install plotly
import plotly.express as px
%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
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
%pip install git+https://github.com/neelnanda-io/neel-plotly.git
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

model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    #refactor_factored_attn_matrices=True,
    device = device,
)

# %%
def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

# %% [markdown]
# # Generate Dataset

# %%


# %%
# generate clean prompts
def generate_clean_corrupted_induction_prompts(model, gen_size = 600, prompt_half_length = 10):
    """
    generates a series of clean and corrupted random token induction prompts
    answer tokens are associated with the correct token in the clean prompt, as well as the correct token in the corrupted prompt
    
    """
    def generate_repeated_tokens(model, batch, seq_len) -> Float[Tensor, "batch seq_len*2"]:
        tokens = torch.randint(1, model.cfg.d_vocab, (batch, seq_len))
        return torch.concat((tokens, tokens), dim=-1)
    


    gen_tokens = generate_repeated_tokens(model, gen_size, prompt_half_length).cuda()
    gen_prompts = [''.join(model.to_str_tokens(gen_tokens[i, :])) for i in range(gen_tokens.shape[0])]

    induction_tokens = []
    induction_prompts = []

    for i in range(gen_size):
    # for prompts that have the same tokens before and after, add to prompts
        if gen_tokens[i].equal(model.to_tokens(gen_prompts[i], prepend_bos=False)[0]):
            induction_tokens.append(gen_tokens[i])
        #induction_prompts.append(gen_prompts[i])

    for i in range(len(induction_tokens)):
    #print(good_tokens[i])
    #print(model.to_tokens(model.to_string(good_tokens[i]), prepend_bos=False))
        assert induction_tokens[i].equal(model.to_tokens(model.to_string(induction_tokens[i]), prepend_bos=False)[0])
    #assert induction_prompts[i] == model.to_string(induction_tokens[i])


    for i, prompt in enumerate(induction_tokens):
    # remove last token
        induction_tokens[i] = prompt[:-1]
    # generate new prompt and add to induction prompts
        induction_prompts.append(model.to_string(induction_tokens[i]))

    broken_batch_size = len(induction_tokens)
    possible_broken_corrupted_tokens = []
    possible_broken_corrupted_prompts = []

    for i in range(broken_batch_size):
        temp = induction_tokens[i].clone()
        temp[prompt_half_length] = torch.randint(1, model.cfg.d_vocab, (1,))
        possible_broken_corrupted_tokens.append(temp)
        possible_broken_corrupted_prompts.append(model.to_string(temp))
    
    BOS_TOKEN = model.to_tokens("")[0].item()

    # filter possible_broken_corrupted_tokens and possible_broken_corrupted_prompts to only include only the ones to which tokenization remains same
    corrupted_tokens = []
    corrupted_prompts = []

    # ALSO ADDS BOS!
    num_removed = 0
    for i in range(broken_batch_size):
        if possible_broken_corrupted_tokens[i].equal(model.to_tokens(possible_broken_corrupted_prompts[i], prepend_bos=False)[0]):
            new_corrupted_token = torch.cat((torch.tensor([BOS_TOKEN]).cuda(), possible_broken_corrupted_tokens[i])).cuda()
            corrupted_tokens.append(new_corrupted_token)
            corrupted_prompts.append(model.to_string(new_corrupted_token))

            new_clean_token = torch.cat((torch.tensor([BOS_TOKEN]).cuda(), induction_tokens[i - num_removed])).cuda()
            induction_tokens[i - num_removed] = new_clean_token
            induction_prompts[i - num_removed] = model.to_string(new_clean_token)
        else:
        # remove associated induction prompt
            induction_tokens.pop(i - num_removed)
            induction_prompts.pop(i - num_removed)
            num_removed += 1

    assert len(corrupted_tokens) == len(corrupted_prompts) == len(induction_tokens) == len(induction_prompts)
    batch_size = len(corrupted_tokens)
    clean_tokens = torch.stack(induction_tokens)
    corrupted_tokens = torch.stack(corrupted_tokens)
    answer_tokens = torch.stack((clean_tokens[:, prompt_half_length + 1], corrupted_tokens[:, prompt_half_length + 1]), dim=1)

    # delete temp variables
    del possible_broken_corrupted_tokens
    del possible_broken_corrupted_prompts
    return corrupted_tokens,batch_size,clean_tokens,answer_tokens

corrupted_tokens, batch_size, clean_tokens, answer_tokens = generate_clean_corrupted_induction_prompts(model, gen_size = 100, prompt_half_length=12)

clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

# %%
answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
correct_residual_direction, incorrect_residual_direction = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_direction - incorrect_residual_direction

# %%
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # SOLUTION
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
    cache: ActivationCache = None,
    logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
    clean_cache = clean_cache,
    use_clean_cache = True,
) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    '''
    batch_size = residual_stack.size(-2)
    if use_clean_cache:
        scaled_residual_stack = clean_cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    else:
        scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)

    return einops.einsum(
        scaled_residual_stack, logit_diff_directions,
        "... batch d_model, batch d_model -> ..."
    ) / batch_size

def calc_all_logit_diffs(cache, use_clean_cache = True):
  clean_per_head_residual, labels = cache.stack_head_results(layer = -1, return_labels = True, apply_ln = False) # per_head_residual.shape = heads batch seq_pos d_model
  # also, for the worried, no, we're not missing the application of LN here since it gets applied in the below function call
  per_head_logit_diff: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, use_clean_cache = use_clean_cache)

  per_head_logit_diff = einops.rearrange(
      per_head_logit_diff,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  correct_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = correct_residual_direction, use_clean_cache = use_clean_cache) 

  correct_direction_per_head_logit = einops.rearrange(
      correct_direction_per_head_logit,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  incorrect_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = incorrect_residual_direction, use_clean_cache = use_clean_cache)

  incorrect_direction_per_head_logit = einops.rearrange(
      incorrect_direction_per_head_logit,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  return per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit

def display_all_logits(cache, title = "Logit Contributions", comparison = False, return_fig = False, logits = None):

  a,b,c = calc_all_logit_diffs(cache)
  if logits is not None:
    ld = logits_to_ave_logit_diff(logits)
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

    ca, cb, cc = calc_all_logit_diffs(clean_cache)
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

def return_item(item):
    return item

def display_corrupted_clean_logits(cache, title = "Logit Contributions", comparison = False, return_fig = False, logits = None):

  a,b,c = calc_all_logit_diffs(cache)
  if logits is not None:
    ld = logits_to_ave_logit_diff(logits)
  else:
    ld = 0.00

  if not comparison:
    fig = imshow(
        torch.stack([a]),
        return_fig = True,
        facet_col = 0,
        facet_labels = [f"Logit Diff - {ld:.2f}"],
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1500,
        margin={"r": 100, "l": 100}
    )
  else:

    ca, cb, cc = calc_all_logit_diffs(clean_cache)
    fig = imshow(
        torch.stack([a, ca, a - ca]),
        return_fig = True,
        facet_col = 0,
        facet_labels = [f"Ablated Logit Differences: {ld:.2f}", f"Clean Logit Differences: {clean_average_logit_diff:.2f}", f"Difference between Ablated and Clea: {(ld - clean_average_logit_diff):.2f}",],
        title=title,
        labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=1000,
        margin={"r": 100, "l": 100}
    )


  if return_fig:
    return fig
  else:
    return a - ca


def stare_at_attention_and_head_pat(cache, layer_to_stare_at, head_to_isolate, display_corrupted_text = False, verbose = True, specific = False, specific_index = 0):
  """
  given a cache from a run, displays the attention patterns of a layer, as well as printing out how much the model
  attends to the S1, S2, and IO token
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

def y_x_logit_diff_plot(per_head_logit_diff, ablated_logit_diff, x_title =  "Clean Direct Effect", y_title = "Ablated Direct Effect", title = "Logit Differences When Sample Ablating in Layer 9 Name Mover Heads"):
    x =  per_head_logit_diff.flatten()
    y =  ablated_logit_diff.flatten()


    fig = px.scatter()
    names = [str((i,j)) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

    fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(), text=names, textposition="top center", mode = 'markers', name = "gpt-2"))

    x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
    fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ))
    fig.update_xaxes(title = x_title)
    fig.update_yaxes(title = y_title)
    fig.update_layout(title = title, width = 950)
    fig.show()

per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit = calc_all_logit_diffs(clean_cache)

# %%
answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)

correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
print(f"Logit difference directions shape:", logit_diff_directions.shape)

# %%

clean_average_logit_diff = logits_to_ave_logit_diff(clean_logits)
corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits)

print(clean_average_logit_diff)
print(corrupted_average_logit_diff)

# %%
diff_from_unembedding_bias = model.b_U[answer_tokens[:, 0]] -  model.b_U[answer_tokens[:, 1]]
final_residual_stream: Float[Tensor, "batch seq d_model"] = clean_cache["resid_post", -1]
print(f"Final residual stream shape: {final_residual_stream.shape}")
final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

print(f"Calculated average logit diff: {(residual_stack_to_logit_diff(final_token_residual_stream, clean_cache, logit_diff_directions) + diff_from_unembedding_bias.mean(0)):.10f}") # <-- okay b_U exists... and matters
print(f"Original logit difference:     {clean_average_logit_diff:.10f}")


# %%
top_heads = []
k = 5

flattened_tensor = per_head_logit_diff.flatten().cpu()
_, topk_indices = torch.topk(flattened_tensor, k)
top_layer_arr, top_index_arr = np.unravel_index(topk_indices.numpy(), per_head_logit_diff.shape)

for l, i in zip(top_layer_arr, top_index_arr):
  top_heads.append((l,i))

print(top_heads)

# %%

fig = display_all_logits(clean_cache, title = "Logit Contributions on Clean Dataset", return_fig = True, logits = clean_logits)


# %%
# Ablate top induction head
heads =  [(7,1)] #[(j,i) for i in range(12) for j in range(6,11)]

model.reset_hooks()
patched_logits = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    patching_metric = return_item,
    verbose = False,
    apply_metric_to_cache = False
)

model.reset_hooks()
ablated_logit_diff, _, _ = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    patching_metric = partial(calc_all_logit_diffs, use_clean_cache = True),
    verbose = False,
    apply_metric_to_cache = True
)

# %%
y_x_logit_diff_plot(per_head_logit_diff, ablated_logit_diff)

# %%

def accumulated_backup_amount(per_head_logit_diff: Float[Tensor, "layer head"], layer, head, clean_per_head_logit_diff: Float[Tensor, "layer head"]):
   diff_in_heads = per_head_logit_diff - clean_per_head_logit_diff
   return diff_in_heads.flatten().sum(0).item() - diff_in_heads[layer, head].item()
   

accumulated_amounts = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for layer in range(model.cfg.n_layers):
   for head in range(model.cfg.n_heads):
        model.reset_hooks()
        ablated_logit_diff, _, _ = act_patch(
            model = model,
            orig_input = clean_tokens,
            new_cache = corrupted_cache,
            patching_nodes = [Node("z", layer = layer, head = head)],
            patching_metric = partial(calc_all_logit_diffs, use_clean_cache = True),
            verbose = False,
            apply_metric_to_cache = True
        )
        accumulated_amounts[layer, head] = accumulated_backup_amount(ablated_logit_diff, layer, head, per_head_logit_diff)


# %%
y_x_logit_diff_plot(per_head_logit_diff, accumulated_amounts, y_title= "accumulated backup", title = "Accumulated Backup Amounts")


# %%
stare_at_attention_and_head_pat(clean_cache,10,0
                                , display_corrupted_text = False, verbose = True, specific = False, specific_index = 0)

# %% [markdown]
# 

# %%
def noising_ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = clean_average_logit_diff,
    corrupted_logit_diff: float = corrupted_average_logit_diff,
) -> float:
    '''
    Given logits, returns how much the performance has been corrupted due to noising.

    We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
    and -1 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    #print(logits[-1, -1])
    patched_logit_diff = logits_to_ave_logit_diff(logits)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()

print(f"IOI metric (IOI dataset): {noising_ioi_metric(clean_logits):.4f}")
print(f"IOI metric (ABC dataset): {noising_ioi_metric(corrupted_logits):.4f}")

def denoising_ioi_metric(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_logit_diff: float = clean_average_logit_diff,
    corrupted_logit_diff: float = corrupted_average_logit_diff,
) -> float:
    '''
    We calibrate this so that the value is 1 when performance got restored (i.e. same as IOI dataset),
    and 0 when performance has been destroyed (i.e. is same as ABC dataset).
    '''
    patched_logit_diff = logits_to_ave_logit_diff(logits)
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff) + 1).item()


print(f"IOI metric (IOI dataset): {denoising_ioi_metric(clean_logits):.4f}")
print(f"IOI metric (ABC dataset): {denoising_ioi_metric(corrupted_logits):.4f}")

# %%
def path_patch_to_head(layer, head, noisily = True):
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
          patching_metric=noising_ioi_metric,
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
          patching_metric=denoising_ioi_metric,
      )['z'])


  return head_results

# %%
# all_path_patch_results = {}

# %%
# for head in range(12):
#     layer = 10
#     results = path_patch_to_head(layer, head)
#     all_path_patch_results[f"{layer}_{head}_noisy"] = results

# %%
layer = 11
head = 3
imshow(
  torch.stack(path_patch_to_head(layer,head)) * 100,
  facet_col = 0,
  facet_labels = ["Query", "Key", "Value", "Pattern"],
  title=f"Effect of Noisily Path Patching in {layer}.{head}",
  labels={"x": "Head", "y": "Layer", "color": "Percent Negative Degretation"},
  coloraxis=dict(colorbar_ticksuffix = "%"),
  border=True,
  width=1500,
  margin={"r": 100, "l": 100},
)

# %% [markdown]
# ## ablation experiments
# i want to see if backup is actually going on lol

# %%
ablate_heads = [(8,i) for i in range(model.cfg.n_heads)]#(7,i) for i in range(12)]
act_logit_results = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = int(layer), head = int(head)) for layer, head in ablate_heads],
    patching_metric = return_item,
    apply_metric_to_cache = False
    )


act_patching_results = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes =  [Node("z", layer = int(layer), head = int(head)) for layer, head in ablate_heads],
    patching_metric = partial(display_all_logits, title = f"Logits when Act Patching in {layer}.{head}", comparison = True, logits = act_logit_results),
    apply_metric_to_cache = True
    )

# %%



# look for previous token heads
def prev_attn_scores(cache: ActivationCache) -> List[str]:
    '''
    Returns a list e.g. ["0.2", "1.4", "1.9"] of "layer.head" which you judge to be prev-token heads
    '''
    prev_attn_head_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for i in range(model.cfg.n_layers):
      attn = cache["pattern", i]

      # if batch dim exists, average across it
      if attn.dim() == 4:
        attn = attn.mean(0)

      seq_len = attn.shape[-1]
      prev_diagonal_mean = attn.diagonal(-1, -2, -1).mean(-1)
      prev_attn_head_scores[i] = prev_diagonal_mean

    return prev_attn_head_scores

# %%
clean_prev_attn_scores = prev_attn_scores(clean_cache)
imshow(
    clean_prev_attn_scores,
    labels={"x":"Head", "y":"Layer"},
    title="(Clean) Prev Token Attention Scores Per Head",
    width=600
)
# %%

def zero_ablation_hook(
    attn_result: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int
) -> Float[Tensor, "batch seq n_heads d_model"]:
    attn_result[:, :, head_index_to_ablate, :] = torch.zeros(attn_result[:, :, head_index_to_ablate, :].shape)
    return attn_result


def mean_ablation_hook(
    attn_result: Float[Tensor, "batch seq n_heads d_model"],
    hook: HookPoint,
    head_index_to_ablate: int,
    mean_results: Float[Tensor, "seq n_heads d_model"]
) -> Float[Tensor, "batch seq n_heads d_model"]:
    attn_result[:, :, head_index_to_ablate, :] = torch.zeros(attn_result[:, :, head_index_to_ablate, :].shape)
    return attn_result

# %%



# %%
def get_backup_from_metric_and_ablations(model, clean_tokens, clean_cache, corrupted_cache, return_item, accumulated_backup_amount, 
                                          zero_ablation_hook, mean_ablation_hook, metric, clean_metric_amounts, get_first_return = False):
    """
    
    get_first_return is a hacky way to get the first return of a function which has multiple returns. this is to work with calc_all_logit_diffs
    """
    
    mean_ablation_prev_token_backup_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        model.reset_hooks()
        temp_hook_fn = partial(mean_ablation_hook, head_index_to_ablate = head, mean_results = clean_cache[utils.get_act_name("z", layer)].mean(0))
        model.add_hook(utils.get_act_name("z", layer), temp_hook_fn)
        logits, cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()


        if get_first_return:
            new_prev_attn_scores = metric(cache)[0]
        else:
            new_prev_attn_scores = metric(cache)

          
        mean_ablation_prev_token_backup_scores[layer, head] = accumulated_backup_amount(new_prev_attn_scores, layer, head, clean_metric_amounts)

    zero_ablation_prev_token_backup_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        model.reset_hooks()
        temp_hook_fn = partial(zero_ablation_hook, head_index_to_ablate = head)
        model.add_hook(utils.get_act_name("z", layer), temp_hook_fn)
        logits, cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()


        if get_first_return:    
            new_prev_attn_scores = metric(cache)[0]
        else:    
            new_prev_attn_scores = metric(cache)
        
        zero_ablation_prev_token_backup_scores[layer, head] = accumulated_backup_amount(new_prev_attn_scores, layer, head, clean_metric_amounts)

    sample_ablation_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
      for head in range(model.cfg.n_heads):
        model.reset_hooks()
   
        cache = act_patch(
            model = model,
            orig_input = clean_tokens,
            new_cache = corrupted_cache,
            patching_nodes =  [Node("z", layer = int(layer), head = int(head))],
            patching_metric = return_item,
            apply_metric_to_cache = True
        )
        model.reset_hooks()
        
        if get_first_return:
            new_prev_attn_scores = metric(cache)[0]
        else:
            new_prev_attn_scores = metric(cache)

        sample_ablation_scores[layer, head] = accumulated_backup_amount(new_prev_attn_scores, layer, head, clean_metric_amounts)
    
    return mean_ablation_prev_token_backup_scores,zero_ablation_prev_token_backup_scores,sample_ablation_scores

mean_ablation_prev_token_backup_scores, zero_ablation_prev_token_backup_scores, sample_ablation_scores = get_backup_from_metric_and_ablations(model, clean_tokens,
                                     clean_cache, corrupted_cache, return_item, accumulated_backup_amount, zero_ablation_hook,
                                       mean_ablation_hook, prev_attn_scores, clean_prev_attn_scores)

# %%

x =  clean_prev_attn_scores.flatten()
# make subplot fig
import plotly
fig = plotly.subplots.make_subplots(rows=1, cols=3, subplot_titles=("Zero Ablation", "Mean Ablation", "Sample Ablation"), shared_yaxes=True)

names = [str((i,j)) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

fig.add_trace(go.Scatter(x = x.cpu(), y = zero_ablation_prev_token_backup_scores.flatten().cpu(), text=names, textposition="top center", mode = 'markers', name = model_name), row = 1, col = 1)
fig.add_trace(go.Scatter(x = x.cpu(), y = mean_ablation_prev_token_backup_scores.flatten().cpu(), text=names, textposition="top center", mode = 'markers', name = model_name), row = 1, col = 2)
fig.add_trace(go.Scatter(x = x.cpu(), y = sample_ablation_scores.flatten().cpu(), text=names, textposition="top center", mode = 'markers', name = model_name), row = 1, col = 3)


x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ), row = 1, col = 1)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ), row = 1, col = 2)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ), row = 1, col = 3)
fig.update_xaxes(title = "clean prev token attention")
fig.update_yaxes(title = "accumulated prev token backup")
fig.update_layout(title = "backup of previous token attention scores under various ablations", width = 1150)
fig.show()


#y_x_logit_diff_plot(clean_prev_attn_scores, prev_token_backup_scores, x_title = "clean prev token score", y_title= "accumulated prev token attention backup", title = "Accumulated Backup Amounts from Mean Ablation")
# %%

# do the same thing but for induction logit diff
mean_ablation_induction_logit_diff_backup_scores, zero_ablation_induction_logit_diff_backup_scores, sample_ablation_induction_logit_diff_backup_scores = get_backup_from_metric_and_ablations(model, clean_tokens,
                                     clean_cache, corrupted_cache, return_item, accumulated_backup_amount, zero_ablation_hook,
                                       mean_ablation_hook, calc_all_logit_diffs, per_head_logit_diff, get_first_return = True)



# %%
# graph the results
x =  per_head_logit_diff.flatten()
# make subplot fig
import plotly
fig = plotly.subplots.make_subplots(rows=1, cols=3, subplot_titles=("Zero Ablation", "Mean Ablation", "Sample Ablation"), shared_yaxes=True)

names = [str((i,j)) for i in range(model.cfg.n_layers) for j in range(model.cfg.n_heads)]

fig.add_trace(go.Scatter(x = x.cpu(), y = zero_ablation_induction_logit_diff_backup_scores.flatten().cpu(), text=names, textposition="top center", mode = 'markers', name = model_name), row = 1, col = 1)
fig.add_trace(go.Scatter(x = x.cpu(), y = mean_ablation_induction_logit_diff_backup_scores.flatten().cpu(), text=names, textposition="top center", mode = 'markers', name = model_name), row = 1, col = 2)
fig.add_trace(go.Scatter(x = x.cpu(), y = sample_ablation_induction_logit_diff_backup_scores.flatten().cpu(), text=names, textposition="top center", mode = 'markers', name = model_name), row = 1, col = 3)


x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ), row = 1, col = 1)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ), row = 1, col = 2)
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ), row = 1, col = 3)
fig.update_xaxes(title = "clean prev logit diff")
fig.update_yaxes(title = "accumulated logit diff backup")
fig.update_layout(title = "backup of logit diff under various ablations", width = 1150)
fig.show()







# %%


layer = 1
head = 3
model.reset_hooks()
temp_hook_fn = partial(zero_ablation_hook, head_index_to_ablate = head)
model.add_hook(utils.get_act_name("z", layer), temp_hook_fn)
logits, cache = model.run_with_cache(clean_tokens)
temp_prev_attn = prev_attn_scores(cache)
model.reset_hooks()

# %%
y_x_logit_diff_plot(clean_prev_attn_scores, temp_prev_attn, y_title= "accumulated prev token attention backup", title = "Accumulated Backup Amounts")
# %%

def get_change_in_head_attn_score(cache, layer, head):
   return prev_attn_scores(cache)[layer, head].item() - clean_prev_attn_scores[layer, head].item()


head_results = []
for i in ["q", "k", "v", "pattern"]:
    temp_results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for r_layer in range(model.cfg.n_layers):
       for r_head in range(model.cfg.n_heads):
        model.reset_hooks() # callum library buggy
        temp_results[r_layer, r_head] = path_patch(
            model,
            orig_input=clean_tokens,
            new_input=corrupted_tokens,
            #new_cache = 'zero',
            sender_nodes=[Node("z", layer = layer, head =head)],
            receiver_nodes=[Node(i, r_layer, r_head)],
            apply_metric_to_cache=True,
            patching_metric=partial(get_change_in_head_attn_score, layer = r_layer, head = r_head),
        )
    head_results.append(temp_results)


# %%
imshow(
  torch.stack(head_results),
  facet_col = 0,
  facet_labels = ["Query", "Key", "Value", "Pattern"],
  title=f"Effect of Noisily Sample Path Patching from {layer}.{head}",
  labels={"x": "Head", "y": "Layer", "color": "Change in Previous Attention Score"},
  border=True,
  width=1500,
  margin={"r": 100, "l": 100},
)
# %%
