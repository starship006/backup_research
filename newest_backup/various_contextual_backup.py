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

#%pip install plotly
import plotly
import plotly.express as px
#%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
import circuitsvis as cv
import os, sys


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

# %%

model_name = "gpt2-small"
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

objects = [
  "perfume",
  "scissors",
  "drum",
  "trumpet",
  "phone",
  "football",
  "token",
  "bracelet",
  "badge",
  "novel",
  "pillow",
  "coffee",
  "skirt",
  "balloon",
  "photo",
  "plate",
  "headphones",
  "flask",
  "menu",
  "compass",
  "belt",
  "wallet",
  "pen",
  "mask",
  "ticket",
  "suitcase",
  "sunscreen",
  "letter",
  "torch",
  "cocktail",
  "spoon",
  "comb",
  "shirt",
  "coin",
  "cable",
  "button",
  "recorder",
  "frame",
  "key",
  "card",
  "canvas",
  "packet",
  "bowl",
  "receipt",
  "pan",
  "report",
  "book",
  "cap",
  "charger",
  "rake",
  "fork",
  "map",
  "soap",
  "cash",
  "whistle",
  "rope",
  "violin",
  "scale",
  "diary",
  "ruler",
  "mouse",
  "toy",
  "cd",
  "dress",
  "shampoo",
  "flashlight",
  "newspaper",
  "puzzle",
  "tripod",
  "brush",
  "cane",
  "whisk",
  "tablet",
  "purse",
  "paper",
  "vinyl",
  "camera",
  "guitar",
  "necklace",
  "mirror",
  "cup",
  "cloth",
  "flag",
  "socks",
  "shovel",
  "cooler",
  "hammer",
  "shoes",
  "chalk",
  "wrench",
  "towel",
  "glove",
  "speaker",
  "remote",
  "leash",
  "magazine",
  "notebook",
  "candle",
  "feather",
  "gloves",
  "mascara",
  "charcoal",
  "pills",
  "laptop",
  "pamphlet",
  "knife",
  "kettle",
  "scarf",
  "tie",
  "goggles",
  "fins",
  "lipstick",
  "shorts",
  "joystick",
  "bookmark",
  "microphone",
  "hat",
  "pants",
  "umbrella",
  "harness",
  "roller",
  "blanket",
  "folder",
  "bag",
  "crate",
  "pot",
  "watch",
  "mug",
  "sandwich",
  "yarn",
  "ring",
  "backpack",
  "glasses",
  "pencil",
  "broom",
  "baseball",
  "basket",
  "loaf",
  "coins",
  "bakery",
  "tape",
  "helmet",
  "bible",
  "jacket"
]

names = [
  " Sebastian",
  " Jack",
  " Jeremiah",
  " Ellie",
  " Sean",
  " William",
  " Caroline",
  " Cooper",
  " Xavier",
  " Ian",
  " Mark",
  " Brian",
  " Carter",
  " Nicholas",
  " Peyton",
  " Luke",
  " Alexis",
  " Ted",
  " Jan",
  " Ty",
  " Jen",
  " Sophie",
  " Kelly",
  " Claire",
  " Leo",
  " Nolan",
  " Kyle",
  " Ashley",
  " Samantha",
  " Avery",
  " Jackson",
  " Hudson",
  " Rebecca",
  " Robert",
  " Joshua",
  " Olivia",
  " Reagan",
  " Lauren",
  " Chris",
  " Chelsea",
  " Deb",
  " Chloe",
  " Madison",
  " Kent",
  " Thomas",
  " Oliver",
  " Dylan",
  " Ann",
  " Audrey",
  " Greg",
  " Henry",
  " Emma",
  " Josh",
  " Mary",
  " Daniel",
  " Carl",
  " Scarlett",
  " Ethan",
  " Levi",
  " Eli",
  " James",
  " Patrick",
  " Isaac",
  " Brooke",
  " Alexa",
  " Eleanor",
  " Anthony",
  " Logan",
  " Damian",
  " Jordan",
  " Tyler",
  " Haley",
  " Isabel",
  " Alan",
  " Lucas",
  " Dave",
  " Susan",
  " Joseph",
  " Brad",
  " Joe",
  " Vincent",
  " Maya",
  " Will",
  " Jessica",
  " Sophia",
  " Angel",
  " Steve",
  " Benjamin",
  " Eric",
  " Cole",
  " Justin",
  " Amy",
  " Nora",
  " Seth",
  " Anna",
  " Stella",
  " Frank",
  " Larry",
  " Alexandra",
  " Ken",
  " Lucy",
  " Katherine",
  " Leah",
  " Adrian",
  " David",
  " Liam",
  " Christian",
  " John",
  " Nathaniel",
  " Andrea",
  " Laura",
  " Kim",
  " Kevin",
  " Colin",
  " Marcus",
  " Emily",
  " Sarah",
  " Steven",
  " Eva",
  " Richard",
  " Faith",
  " Amelia",
  " Harper",
  " Keith",
  " Ross",
  " Megan",
  " Brooklyn",
  " Tom",
  " Grant",
  " Savannah",
  " Riley",
  " Julia",
  " Piper",
  " Wyatt",
  " Jake",
  " Nathan",
  " Nick",
  " Blake",
  " Ryan",
  " Jason",
  " Chase",]
saved_names = names

places = [
  "swamp",
  "school",
  "volcano",
  "hotel",
  "subway",
  "arcade",
  "library",
  "island",
  "convent",
  "pool",
  "mall",
  "prison",
  "quarry",
  "temple",
  "ruins",
  "factory",
  "zoo",
  "mansion",
  "tavern",
  "planet",
  "forest",
  "airport",
  "pharmacy",
  "church",
  "park",
  "delta",
  "mosque",
  "valley",
  "casino",
  "pyramid",
  "aquarium",
  "castle",
  "ranch",
  "clinic",
  "theater",
  "gym",
  "studio",
  "station",
  "palace",
  "stadium",
  "museum",
  "plateau",
  "home",
  "resort",
  "garage",
  "reef",
  "lounge",
  "chapel",
  "canyon",
  "brewery",
  "market",
  "jungle",
  "office",
  "cottage",
  "street",
  "gallery",
  "landfill",
  "glacier",
  "barracks",
  "bakery",
  "synagogue",
  "jersey",
  "plaza",
  "garden",
  "cafe",
  "cinema",
  "beach",
  "harbor",
  "circus",
  "bridge",
  "monastery",
  "desert",
  "tunnel",
  "motel",
  "fortress"
]
# %%
 
one_token_no_space_names = [] # names for which the no-space, lower case version of it is also a single token
for name in names:
    shortened_name = name.lower().replace(" ", "")
    if len(model.to_str_tokens(shortened_name, prepend_bos=False)) == 1:
        one_token_no_space_names.append(name)

lower_case_still_same_name = []
for name in names:
    lower_name = name.lower()
    if len(model.to_str_tokens(lower_name, prepend_bos=False)) == 1:
        lower_case_still_same_name.append(name)

one_token_objects = []
for obj in objects:
    longer_obj = " " + obj
    if len(model.to_str_tokens(longer_obj, prepend_bos=False)) == 1:
        one_token_objects.append(longer_obj)

one_token_places = []
for place in places:
    longer_place = " " + place
    if len(model.to_str_tokens(longer_place, prepend_bos=False)) == 1:
        one_token_places.append(longer_place)

one_token_names = []

for name in names:
    longer_name = name
    if len(model.to_str_tokens(longer_name, prepend_bos=False)) == 1:
        one_token_names.append(longer_name)

# %%
type_names = ["IOI", "Induction", "Handle Filling", "Mr/Mrs Implication"]
IOI_template = "When{name_A} and{name_B} went to the{place},{name_C} gave the{object} to"

# test variants of induction
Induction_templates = ["So,{name_A} ensured the{object} was brought out of the{place}. The",
                       "Thus,{name_A} allowed the{object} to enter into the grand{place}. The",
                       "Now,{name_A} guaranteed the{object} had fallen into the nearby{place}. The"]

handle_templates = ["To learn more about the{object}, we spoke with{name_A}{name_B} (@{modified_name_A}",
                   "To build our own{object}, we got help from{name_A}{name_B} (@{modified_name_A}"]

ignore_mr_templates = ["We talked to{name_A}{name_B}, who held the{object}. As Mr.",
                       "We spoke with{name_A}{name_B}, who held the{object}. As Mrs."]
# %%
ioi_string = IOI_template.format(
    name_A = one_token_names[0],
    name_B = one_token_names[1],
    name_C = one_token_names[0],
    place = one_token_places[0],
    object = one_token_objects[0]
)
induction_string_A = Induction_templates[0].format(
    name_A = one_token_names[1],
    object = one_token_objects[2],
    place = one_token_places[5]
)
induction_string_B = Induction_templates[1].format(
    name_A = one_token_names[4],
    object = one_token_objects[1],
    place = one_token_places[3]
)
induction_string_C = Induction_templates[2].format(
    name_A = one_token_names[2],
    object = one_token_objects[3],
    place = one_token_places[2]
)

handle_string_A = handle_templates[0].format(
    object = one_token_objects[0],
    name_A = one_token_no_space_names[4],
    name_B = one_token_no_space_names[1],
    # modified name is name_A but all lower case and now leading space
    modified_name_A = one_token_no_space_names[4].lower().replace(" ", "")
)

handle_string_B = handle_templates[1].format(
    object = one_token_objects[1],
    name_A = one_token_no_space_names[2],
    name_B = one_token_no_space_names[3],
    # modified name is name_A but all lower case and now leading space
    modified_name_A = one_token_no_space_names[2].lower().replace(" ", "")
)

ignore_mr_string = ignore_mr_templates[0].format(
    name_A = one_token_names[0],
    name_B = one_token_names[1],
    object = one_token_objects[0]
)

ignore_mrs_string = ignore_mr_templates[1].format(
    name_A = one_token_names[0],
    name_B = one_token_names[1],
    object = one_token_objects[0]
)



# %%
print(ioi_string)
print(induction_string_A)
print(induction_string_B)
print(induction_string_C)
print(handle_string_A)
print(handle_string_B)
print(ignore_mr_string)
print(ignore_mrs_string)


print(model.to_tokens(ioi_string).shape)
print(model.to_tokens(induction_string_A).shape)
print(model.to_tokens(induction_string_B).shape)
print(model.to_tokens(induction_string_C).shape)
print(model.to_tokens(handle_string_A).shape)
print(model.to_tokens(handle_string_B).shape)
print(model.to_tokens(ignore_mr_string).shape)
print(model.to_tokens(ignore_mrs_string).shape)

print(model.to_str_tokens(ioi_string))
print(model.to_str_tokens(induction_string_A))
print(model.to_str_tokens(induction_string_B))
print(model.to_str_tokens(induction_string_C))
print(model.to_str_tokens(handle_string_A))
print(model.to_str_tokens(handle_string_B))
print(model.to_str_tokens(ignore_mr_string))
print(model.to_str_tokens(ignore_mrs_string))
# %%

TOTAL_TYPES = 4
NUM_PROMPTS = 15 * 6 * TOTAL_TYPES
PROMPTS_PER_TYPE = int(NUM_PROMPTS / TOTAL_TYPES)


def generate_dataset(NUM_PROMPTS, TOTAL_TYPES):
    PROMPTS = []
    CORRUPTED_PROMPTS = []
    ANSWERS = []
    INCORRECT_ANSWERS = []
    # generate IOI prompts
    for i in range(PROMPTS_PER_TYPE):
        name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
        # generate name B that is different than A
        name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_B == name_A:
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]

        #generate name C diff from A and B
        name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_C == name_A or name_C == name_B:
            name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]

        # generate diff name D
        name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]
        while name_D == name_A or name_D == name_B or name_D == name_C:
            name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]

        place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
        object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]

        PROMPTS.append(IOI_template.format(
            name_A = name_A,
            name_B = name_B,
            name_C = name_A,
            place = place_A,
            object = object_A
        ))

        CORRUPTED_PROMPTS.append(IOI_template.format(
            name_A = name_C,
            name_B = name_D,
            name_C = name_C,
            place = place_A,
            object = object_A
        ))

        ANSWERS.append(name_B)
        INCORRECT_ANSWERS.append(name_A)


    # generate induction prompts
    INDUCTION_TYPES = 3
    for i in range(int(PROMPTS_PER_TYPE / INDUCTION_TYPES)):
        for prompt_type in range(INDUCTION_TYPES):
            name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]

            object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
            object_B = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
            while object_B == object_A:
                object_B = one_token_objects[random.randint(0, len(one_token_objects) - 1)]


            place_A = one_token_places[random.randint(0, len(one_token_places) - 1)]
            PROMPTS.append(Induction_templates[prompt_type].format(
                name_A = name_A,
                object = object_A,
                place = place_A
            ))

            CORRUPTED_PROMPTS.append(Induction_templates[prompt_type].format(
                name_A = name_A,
                object = object_B,
                place = place_A
            ))

            ANSWERS.append(object_A)
            #INCORRECT_ANSWERS.append(object_B) # if we want incorrect to be some other object
            INCORRECT_ANSWERS.append(name_A) # if we want incorrect to be name in prompt

    # generate handle prompts
    HANDLE_TYPES = 2
    for i in range(int(PROMPTS_PER_TYPE / HANDLE_TYPES)):
        for prompt_type in range(HANDLE_TYPES):
            name_A = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            name_B = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            while name_B == name_A:
                name_B = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]

            name_C = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            while name_C == name_A or name_C == name_B:
                name_C = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            name_D = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]
            while name_D == name_A or name_D == name_B or name_D == name_C:
                name_D = one_token_no_space_names[random.randint(0, len(one_token_no_space_names) - 1)]

        

            object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]
            PROMPTS.append(handle_templates[prompt_type].format(
                object = object_A,
                name_A = name_A,
                name_B = name_B,
                # modified name is name_A but all lower case and now leading space
                modified_name_A = name_A.lower().replace(" ", "")
            ))
        
        
            CORRUPTED_PROMPTS.append(handle_templates[prompt_type].format(
                object = object_A,
                name_A = name_C,
                name_B = name_D,
                # modified name is name_A but all lower case and now leading space
                modified_name_A = name_C.lower().replace(" ", "")
            ))


            ANSWERS.append(name_B.lower().replace(" ", ""))
            INCORRECT_ANSWERS.append(name_A.lower().replace(" ", ""))


    # generate ignore mr prompts
    IGNORE_MR_TYPES = 2
    for i in range(int(PROMPTS_PER_TYPE / IGNORE_MR_TYPES)):
        for prompt_type in range(IGNORE_MR_TYPES):
            name_A = one_token_names[random.randint(0, len(one_token_names) - 1)]
            name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
            while name_B == name_A:
                name_B = one_token_names[random.randint(0, len(one_token_names) - 1)]
            name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]
            while name_C == name_A or name_C == name_B:
                name_C = one_token_names[random.randint(0, len(one_token_names) - 1)]
            name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]
            while name_D == name_A or name_D == name_B or name_D == name_C:
                name_D = one_token_names[random.randint(0, len(one_token_names) - 1)]

            object_A = one_token_objects[random.randint(0, len(one_token_objects) - 1)]

            PROMPTS.append(ignore_mr_templates[prompt_type].format(
                name_A = name_A,
                name_B = name_B,
                object = object_A
            ))
            CORRUPTED_PROMPTS.append(ignore_mr_templates[prompt_type].format(
                name_A = name_C,
                name_B = name_D,
                object = object_A
            ))

            ANSWERS.append(name_B)
            INCORRECT_ANSWERS.append(name_A)
    return PROMPTS, CORRUPTED_PROMPTS, ANSWERS, INCORRECT_ANSWERS

PROMPTS, CORRUPT_PROMPTS, ANSWERS, INCORRECT_ANSWERS = generate_dataset(NUM_PROMPTS, TOTAL_TYPES)


# %%
clean_tokens = model.to_tokens(PROMPTS)
corrupt_tokens = model.to_tokens(CORRUPT_PROMPTS)
answer_tokens = model.to_tokens(ANSWERS, prepend_bos=False)
incorrect_answer_tokens = model.to_tokens(INCORRECT_ANSWERS, prepend_bos=False)
# combine answer and incorrect 
full_answer_tokens = torch.cat((answer_tokens, incorrect_answer_tokens), 1)
# %%
assert clean_tokens.shape[0] == answer_tokens.shape[0] == corrupt_tokens.shape[0] == incorrect_answer_tokens.shape[0] == full_answer_tokens.shape[0] == NUM_PROMPTS
assert answer_tokens.shape[1] == 1
assert full_answer_tokens.shape[1] == 2
# %%
INCLUDE_INCORRECT = False
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupt_logits, corrupted_cache = model.run_with_cache(corrupt_tokens)
logit_directions = model.tokens_to_residual_directions(answer_tokens)[:, 0, :]
logit_diff_directions = model.tokens_to_residual_directions(full_answer_tokens)[:, 0, :] - model.tokens_to_residual_directions(full_answer_tokens)[:, 1, :]

print(utils.lm_accuracy(clean_logits, clean_tokens))
print(utils.lm_cross_entropy_loss(clean_logits, clean_tokens))




# %% Helper Functions

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 1"] = answer_tokens,
    per_prompt: bool = False,
    include_incorrect = False,
    full_answers = full_answer_tokens,
):
    '''
    Returns logit contributions to the correct answers

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]

    if include_incorrect:
        answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=full_answers.to(device))
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        

    else:
        # Get the logits corresponding to the indirect object / subject tokens respectively
        answer_logits: Float[Tensor, "batch 1"] = final_logits.gather(dim=-1, index=answer_tokens.to(device))
        # Find logit difference
        correct_logits = answer_logits.unbind(dim=-1)[0]
        answer_logit_diff = correct_logits



    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def final_logit_to_probabilities( logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 1"] = answer_tokens,):
    '''
    Returns logit contributions to the correct answers
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Convert to probabilities
    final_probs = F.softmax(final_logits, dim=-1)
    # Get probability of correct answer
    answer_probs: Float[Tensor, "batch 1"] = final_probs.gather(dim=-1, index=answer_tokens.to(device))
    return answer_probs

def residual_stack_to_logit_diff(
    residual_stack: Float[Tensor, "... batch d_model"],
    cache_for_ln: ActivationCache = clean_cache,
    logit_directions: Float[Tensor, "batch d_model"] = logit_directions,
    per_prompt = False,
    include_incorrect = False,
    logit_diff_directions = logit_diff_directions,

) -> Float[Tensor, "..."]:
    '''
    Gets the avg logit difference between the correct and incorrect answer for a given
    stack of components in the residual stream.
    '''
    batch_size = residual_stack.size(-2)
    scaled_residual_stack = cache_for_ln.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
    
    direction = logit_diff_directions if include_incorrect else logit_directions
    if per_prompt:
         return einops.einsum(
            scaled_residual_stack, direction,
            "... batch d_model, batch d_model -> ... batch"
        ) 
    else:
        return einops.einsum(
            scaled_residual_stack, direction,
            "... batch d_model, batch d_model -> ..."
        ) / batch_size

def calc_all_logit_contibutions(cache, per_prompt = False, include_incorrect = False) -> Float[Tensor, "layer head"]:
  clean_per_head_residual, labels = cache.stack_head_results(layer = -1, return_labels = True, apply_ln = False) # per_head_residual.shape = heads batch seq_pos d_model
  # also, for the worried, no, we're not missing the application of LN here since it gets applied in the below function call
  per_head_logit_diff: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], clean_cache, per_prompt=per_prompt, include_incorrect=include_incorrect)

  per_head_logit_diff = einops.rearrange(
      per_head_logit_diff,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )
  return per_head_logit_diff

def display_all_logits(cache = None, per_head_ld = None, title = "Logit Contributions", comparison = False,
                        return_fig = False, logits = None, include_incorrect = False):
    """
    given an input, display the logit contributions of each head
    """
    if per_head_ld is not None and cache is None:
        assert per_head_ld.shape == (model.cfg.n_layers, model.cfg.n_heads, NUM_PROMPTS)
    if per_head_ld is not None and cache is not None:
        # throw error - only one should be passed
        raise ValueError("Only one of per_head_ld and cache should be passed")

    if logits is not None:
        ld = logits_to_ave_logit_diff(logits, include_incorrect=include_incorrect)
    else:
        ld = 0.00

    if per_head_ld is not None:
        a = per_head_ld
    else:
        a = calc_all_logit_contibutions(cache, per_prompt=True, include_incorrect=include_incorrect)
    if not comparison:
        a_1, a_2, a_3, a_4 = a[..., :PROMPTS_PER_TYPE], a[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], a[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], a[..., 3*PROMPTS_PER_TYPE:]
        fig = imshow(
            torch.stack([(a_1).mean(-1), 
                            (a_2).mean(-1), (a_3).mean(-1), (a_4).mean(-1)]),
            return_fig = True,
            facet_col = 0,
            facet_labels = type_names,
            title=title,
            labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
            #coloraxis=dict(colorbar_ticksuffix = "%"),
            border=True,
            width=1500,
            margin={"r": 100, "l": 100},
        )
    else:
        ca = calc_all_logit_contibutions(clean_cache, per_prompt = True, include_incorrect=include_incorrect)
        a_1, a_2, a_3, a_4 = a[..., :PROMPTS_PER_TYPE], a[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], a[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], a[..., 3*PROMPTS_PER_TYPE:]
        ca_1, ca_2, ca_3, ca_4 = ca[..., :PROMPTS_PER_TYPE], ca[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], ca[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], ca[..., 3*PROMPTS_PER_TYPE:]

        #print(a_1.shape, ca_1.shape, a_2.shape, ca_2.shape, a_3.shape, ca_3.shape, a_4.shape, ca_4.shape)
        assert a_1.shape == ca_1.shape == a_2.shape == ca_2.shape == a_3.shape == ca_3.shape == a_4.shape == ca_4.shape

        fig = imshow(
            torch.stack([(a_1 - ca_1).mean(-1), 
                            (a_2-ca_2).mean(-1), (a_3 - ca_3).mean(-1), (a_4 - ca_4).mean(-1)]),
            return_fig = True,
            facet_col = 0,
            facet_labels = type_names,
            title=title,
            labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
            #coloraxis=dict(colorbar_ticksuffix = "%"),
            border=True,
            width=1500,
            margin={"r": 100, "l": 100}
        )

    if return_fig:
        return fig
    else:
        fig.show()

def return_item(item):
    return item


# %%
clean_per_prompt_logits = logits_to_ave_logit_diff(clean_logits, answer_tokens, per_prompt=False, include_incorrect=INCLUDE_INCORRECT)
corrupt_per_prompt_logits = logits_to_ave_logit_diff(corrupt_logits, answer_tokens, per_prompt=False, include_incorrect=INCLUDE_INCORRECT)
clean_per_prompt_head_logit_diff: Float[Tensor, "layer head batch"] = calc_all_logit_contibutions(clean_cache, per_prompt=True, include_incorrect=INCLUDE_INCORRECT)
print(clean_per_prompt_logits)
print(corrupt_per_prompt_logits)
# %%
display_all_logits(cache = clean_cache, title = "Logit Contributions on Clean Dataset", return_fig = False, logits = clean_logits, include_incorrect = INCLUDE_INCORRECT)
# %% Ablate the NMHs!

def dir_effects_from_sample_ablating_head(ablate_heads):
    """this function gets the new cache of all the heads when sample ablating the input head
    it uses the global cache, owt_tokens, corrupted_owt_tokens
    """

    
    new_cache = act_patch(model, clean_tokens, [Node("z", layer, head) for (layer,head) in ablate_heads],
                            return_item, corrupt_tokens, apply_metric_to_cache= True)
    return new_cache


def dir_effects_from_path_patching_into_head(ablate_heads, method = "sample", include_incorrect = False) -> Float[Tensor, "layer head batch"]:
    """this function gets the new direct effect of all the heads when path patching into downstream heads
    via sample ablations 
    it uses the global cache, owt_tokens, corrupted_owt_tokens
    """

    
    # new_cache = act_patch(model, clean_tokens, [Node("z", layer, head) for (layer,head) in ablate_heads],
    #                         return_item, corrupt_tokens, apply_metric_to_cache= True)

    results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, NUM_PROMPTS)).cuda()

    for receiver_layer in range(model.cfg.n_layers):
        for receiver_head in range(model.cfg.n_heads):
            per_head_logit_diff = path_patch(model, clean_tokens, corrupt_tokens,
                                    [Node("z", layer, head) for (layer,head) in ablate_heads],
                                    [Node("q", receiver_layer, receiver_head)],
                                     partial(calc_all_logit_contibutions, per_prompt=True, include_incorrect=include_incorrect),
                                     apply_metric_to_cache=True)
            results[receiver_layer, receiver_head] = per_head_logit_diff[receiver_layer, receiver_head] 
            
    return results

# %%
def path_patch_to_and_display(heads, include_incorrect = False):
    path_patch_from_results = dir_effects_from_path_patching_into_head(heads, include_incorrect=include_incorrect)
    display_all_logits(per_head_ld = path_patch_from_results, comparison=True, title = f"Logit Diff Diff of each Head upon Noisily Path Patching into Query from {heads} in {model_name}", include_incorrect=include_incorrect)

# %%
path_patch_to_and_display([[9,6]], include_incorrect=INCLUDE_INCORRECT)
# %%
#path_patch_to_and_display([[9,6], [9,9]])


display_all_logits(dir_effects_from_sample_ablating_head([[9,6], [9,9]]), title = "Sample ablating in 9.9 and 9.6, Logit Diffs", comparison=True, include_incorrect=INCLUDE_INCORRECT)


# %% function that looks at the backup activation on a specific prompt
def backup_activation_on_prompt(prompt_index, cache = None, per_head_contribution = None, include_incorrect = False):
    prompt = PROMPTS[prompt_index]
    answer = ANSWERS[prompt_index]
    if cache is not None and per_head_contribution is not None:
        raise ValueError("Only one of cache and logits should be passed")
    
    if per_head_contribution is None:
        per_head_contribution = calc_all_logit_contibutions(cache, per_prompt=True, include_incorrect=include_incorrect)

    print("PROMPT: " + prompt)
    imshow(per_head_contribution[..., prompt_index] - clean_per_prompt_head_logit_diff[..., prompt_index], title = "Logit Contributions",  labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
 return_fig = False)
    

# %%

new_cache = dir_effects_from_sample_ablating_head([[9,6]])
backup_activation_on_prompt(2, new_cache, include_incorrect=INCLUDE_INCORRECT)
# %%

def show_per_prompt_activation_sliders(heads, include_incorrect = False):
    # calculate the new per_head_logit_diff for ablating each head
    per_head_logit_diffs = []
    for head in heads:
        new_cache = dir_effects_from_sample_ablating_head([head])
        per_head_logit_diffs.append(calc_all_logit_contibutions(new_cache, per_prompt=True, include_incorrect=include_incorrect))
        

    # create a plotly graph that shows all the backup from each head, controlled by a slider that allows you to select which prompt to look at
    
    # make subplots
    fig = make_subplots(rows=1, cols=len(per_head_logit_diffs), shared_yaxes=True, shared_xaxes=True,
                        subplot_titles=[f"Head {head}" for head in heads])

    print(per_head_logit_diffs[0].shape)
    for ablated_head_index in range(len(per_head_logit_diffs)):
        for prompt_index in range(NUM_PROMPTS):
            fig.add_trace(go.Heatmap(
                z = per_head_logit_diffs[ablated_head_index][..., prompt_index].cpu() - clean_per_prompt_head_logit_diff[..., prompt_index].cpu(),
                name = PROMPTS[prompt_index],
                visible = False,
                colorbar = dict(
                                        title = 'Logit Diff Diff',)
            ), row=1, col=ablated_head_index+1)
    
    steps = []
    for i in range(int(len(fig.data) / (len(per_head_logit_diffs)))):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Prompt: " + PROMPTS[i % NUM_PROMPTS]}],  # layout attribute
        )
        for j in range(len(per_head_logit_diffs)):
            step["args"][0]["visible"][(i + NUM_PROMPTS * j) % (NUM_PROMPTS * len(per_head_logit_diffs))] = True
        
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Prompt: "},
        pad={"t": 50},
        steps=steps
    )]
    fig.update_layout(
        sliders=sliders,
        width = 1100
    )

    # change color to be RgBU
    fig.update_traces(
        colorscale = "RdBu",
        # center mid to 0
        zmid = 0,
        # set min and max to be symmetric
        zmin = -2.5,
        zmax = 2.5,
    )
    
    for j in range(len(per_head_logit_diffs)):
        fig.data[0 + j * NUM_PROMPTS].visible = True
    



    fig.update_yaxes(
                    mirror = True,
    )
    fig['layout']['yaxis']['autorange'] = "reversed"

    fig.show()
    return fig




# %%
fig = show_per_prompt_activation_sliders([[9,6], [9,9], [10,0]], include_incorrect = INCLUDE_INCORRECT)
# %%
# save the fig to html

fig.write_html(f"{model_name}_per_prompt_activation_sliders.html")


# %%