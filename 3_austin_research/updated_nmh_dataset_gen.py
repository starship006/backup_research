# %%
import random
import transformer_lens

from transformer_lens import HookedTransformer

from typing import Union, List, Optional
import warnings
import torch as t
import numpy as np
import random
import copy
import re


# %%
# model = HookedTransformer.from_pretrained(
#     "gpt2-small",
#     center_unembed = True, 
#     center_writing_weights = True,
#     fold_ln = True, # TODO; understand this
#     refactor_factored_attn_matrices = False,
# )

# %%

# verbs in past tense
VERBS = [
    "handed",
    "tossed",
    "rolled",
    "passed",
    "gave"
]

OBJECTS = [
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

NAMES = [
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

PLACES = [
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

# %%
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)

set_global_seed(0)
# %%
ABBA_TEMPLATES = [
    "Then,{A} and{B} went to the {PLACE}.{B} {VERB} a {OBJECT} to{A}",
    "Afterwards,{A} and{B} went to the {PLACE}.{B} {VERB} a {OBJECT} to{A}",
    "When{A} and{B} went to the {PLACE},{B} {VERB} a {OBJECT} to{A}",
    "Friends{A} and{B} went to the {PLACE}.{B} {VERB} a {OBJECT} to{A}",
    "As{A} and{B} went to the {PLACE},{B} {VERB} a {OBJECT} to{A}",
    "Today,{A} and{B} strolled in the {PLACE}.{B} {VERB} a {OBJECT} to{A}",
    "As{A} and{B} walked around in the {PLACE},{B} {VERB} a {OBJECT} to{A}",
]

BAAB_TEMPLATES = [template.replace("{A}", "{C}").replace("{B}", "{A}").replace("{C}", "{B}") for template in ABBA_TEMPLATES]

# %%
def get_rand_item_unlike_prev(list_to_pull_from:list, prev_items: set):
    assert len(set(list_to_pull_from)) != len(prev_items)
    assert len(list_to_pull_from) > 0

    while True:
        item = list_to_pull_from[random.randint(0, len(list_to_pull_from) - 1)]
        if item not in prev_items:
            return item

# %%
def make_ioi_prompts(model, template_list: list, name_a_is_correct: bool, BATCH_SIZE = 50):
    # get a list of names that are one token only
    ONE_TOKEN_NAMES = []
    for name in NAMES:
        if len(model.to_tokens(name, prepend_bos = False)[0]) == 1:
            ONE_TOKEN_NAMES.append(name)

    assert model.to_tokens(ONE_TOKEN_NAMES, prepend_bos = False).shape[-1] == 1

    
    
    PROMPTS = []
    ANSWERS = []
    ANSWER_INDICIES = []

    for i in range(BATCH_SIZE):
        unfilled_template = get_rand_item_unlike_prev(template_list, set())

        NAME_A = get_rand_item_unlike_prev(ONE_TOKEN_NAMES, set())
        NAME_B = get_rand_item_unlike_prev(ONE_TOKEN_NAMES, set(NAME_A))
        VERB = get_rand_item_unlike_prev(VERBS, set())
        OBJECT = get_rand_item_unlike_prev(OBJECTS, set())
        PLACE = get_rand_item_unlike_prev(PLACES, set())

        template = unfilled_template.format(
            A = NAME_A,
            B = NAME_B,
            VERB = VERB,
            OBJECT = OBJECT,
            PLACE = PLACE
        )
        correct_name = NAME_A if name_a_is_correct else NAME_B

        # check that the final token is the prediction of the token (this is what we will be measuring)
        tokens = model.to_tokens(template, prepend_bos = False)[0]
        assert tokens[-1] == model.to_tokens(correct_name, prepend_bos = False)[0]

        # add prompts and answers
        PROMPTS.append(template)
        ANSWERS.append(correct_name)
        ANSWER_INDICIES.append(len(tokens) - 1 + 1) # +1 for the future BOS token that gets appended

    return PROMPTS, ANSWERS, ANSWER_INDICIES

# %%
#ABBA_PROMPTS, ABBA_ANSWERS, ABBA_ANSWER_INDICIES = make_ioi_prompts(ABBA_TEMPLATES, True)
#BAAB_PROMPTS, BAAB_ANSWERS, BAAB_ANSWER_INDICIES = make_ioi_prompts(BAAB_TEMPLATES, False)
    


# %%
ignore_mr_templates = ["We talked to{A}{B}, who {VERB} the {OBJECT}. As Mr.{B}",
                        "We spoke with{A}{B}, who {VERB} a {OBJECT}. As Mrs.{B}",
                        "We chatted with{A}{B}, who {VERB} a {OBJECT}. As Ms.{B}",
                        "They talked to{A}{B}, who {VERB} the {OBJECT}. As Dr.{B}",
                        "I conversed with{A}{B}, who {VERB} the {OBJECT}. As Miss{B}",
                        "I talked with{A}{B}, who {VERB} the {OBJECT}. As Mister{B}"]
# %%
#MR_PROMPTS, MR_ANSWERS, MR_ANSWER_INDICIES = make_ioi_prompts(ignore_mr_templates, False)
# %%


def generate_ioi_mr_prompts(model, SUBSEC_SIZE = 50):
    """
    generates SUBSEC_SIZE * 4 prompts, with 1/2 of them being MR/MRS prompts, 1/4 being ABBA prompts, and 1/4 being BAAB prompts

    returns prompts, answers, and answer indicies
    """
    ABBA_PROMPTS, ABBA_ANSWERS, ABBA_ANSWER_INDICIES = make_ioi_prompts(model, ABBA_TEMPLATES, True, BATCH_SIZE=SUBSEC_SIZE)
    BAAB_PROMPTS, BAAB_ANSWERS, BAAB_ANSWER_INDICIES = make_ioi_prompts(model, BAAB_TEMPLATES, False, BATCH_SIZE=SUBSEC_SIZE)
    MR_PROMPTS, MR_ANSWERS, MR_ANSWER_INDICIES = make_ioi_prompts(model, ignore_mr_templates, False, BATCH_SIZE=SUBSEC_SIZE * 2)

    return ABBA_PROMPTS + BAAB_PROMPTS + MR_PROMPTS, ABBA_ANSWERS + BAAB_ANSWERS + MR_ANSWERS, ABBA_ANSWER_INDICIES + BAAB_ANSWER_INDICIES + MR_ANSWER_INDICIES