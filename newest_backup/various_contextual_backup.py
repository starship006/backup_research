# %%

# !sudo apt install unzip
# !pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
# !pip install git+https://github.com/neelnanda-io/neel-plotly.git
# !pip install 
!pip install plotly fancy_einsum jaxtyping transformers datasets transformer_lens
from imports import *
# %%

model_name = "gpt2-small"
backup_storage_file_name = model_name + "_new_backup_count_storage.pickle"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
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
NUM_PROMPTS = 4 * 6 * TOTAL_TYPES
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
    include_incorrect = INCLUDE_INCORRECT,
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
    return ("error not ready")
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
    include_incorrect = INCLUDE_INCORRECT,
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

def calc_all_logit_contibutions(cache, per_prompt = False, include_incorrect = INCLUDE_INCORRECT) -> Float[Tensor, "layer head"]:
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
                        return_fig = False, logits = None, include_incorrect = INCLUDE_INCORRECT):
    """
    given an input, display the logit contributions of each head

    comparison: if True, display logit contribution/diff diff; if False, display logits contibution/diff
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
model.set_use_attn_result(True)
clean_per_prompt_logits = logits_to_ave_logit_diff(clean_logits, answer_tokens, per_prompt=False, include_incorrect=INCLUDE_INCORRECT)
corrupt_per_prompt_logits = logits_to_ave_logit_diff(corrupt_logits, answer_tokens, per_prompt=False, include_incorrect=INCLUDE_INCORRECT)
clean_per_prompt_head_logit_diff: Float[Tensor, "layer head batch"] = calc_all_logit_contibutions(clean_cache, per_prompt=True, include_incorrect=INCLUDE_INCORRECT)
per_head_logit_diff = calc_all_logit_contibutions(clean_cache, per_prompt=False, include_incorrect=INCLUDE_INCORRECT)
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


def dir_effects_from_path_patching_into_head(ablate_heads, method = "sample", include_incorrect = INCLUDE_INCORRECT) -> Float[Tensor, "layer head batch"]:
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
def path_patch_to_and_display(heads, include_incorrect = INCLUDE_INCORRECT):
    path_patch_from_results = dir_effects_from_path_patching_into_head(heads, include_incorrect=include_incorrect)
    display_all_logits(per_head_ld = path_patch_from_results, comparison=True, title = f"Logit Diff Diff of each Head upon Noisily Path Patching into Query from {heads} in {model_name}", include_incorrect=include_incorrect)

# %%
path_patch_to_and_display([[9,9]], include_incorrect=INCLUDE_INCORRECT)
# %%
#path_patch_to_and_display([[9,6], [9,9]])


display_all_logits(dir_effects_from_sample_ablating_head([[9,6], [9,9]]), title = "Sample ablating in 9.9 and 9.6, Logit Diffs", comparison=True, include_incorrect=INCLUDE_INCORRECT)


# %% function that looks at the backup activation on a specific prompt
def backup_activation_on_prompt(prompt_index, cache = None, per_head_contribution = None, include_incorrect = INCLUDE_INCORRECT):
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

def show_per_prompt_activation_sliders(heads, include_incorrect = INCLUDE_INCORRECT):
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

# Query Intervention to isolate some important directions.

def store_activation(
    activation,
    hook: HookPoint,
    where_to_store
):
    """
    takes a storage container where_to_store, and stores the activation in it at a hook
    """""
    where_to_store[:] = activation


def get_projection(from_vector, to_vector):
    dot_product = einops.einsum(from_vector, to_vector, "batch d_model, batch d_model -> batch")
    #print("Average Dot Product of Output Across Batch: " + str(dot_product.mean(0)))
    length_of_from_vector = einops.einsum(from_vector, from_vector, "batch d_model, batch d_model -> batch")
    length_of_vector = einops.einsum(to_vector, to_vector, "batch d_model, batch d_model -> batch")
    projected_lengths = (dot_product) / (length_of_vector)
    #print( einops.repeat(projected_lengths, "batch -> batch d_model", d_model = model.cfg.d_model)[0])
    projections = to_vector * einops.repeat(projected_lengths, "batch -> batch d_model", d_model = to_vector.shape[-1])
    return projections

def project_vector_operation(
    original_resid_stream: Float[Tensor, "batch seq head_idx d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "batch d_model"],
    position = -1,
    heads = [], # array of ints
    scale_proj = 1,
    project_only = False
) -> Float[Tensor, "batch n_head pos pos"]:
  '''
  Function which gets orthogonal projection of residual stream to a vector, and either subtracts it or keeps only it
  '''
  for head in heads:
    projections = get_projection(original_resid_stream[:, position, head, :], vector)
    if project_only:
      original_resid_stream[:, position, head, :] = projections * scale_proj
    else:
      original_resid_stream[:, position, head, :] = (original_resid_stream[:, position, head, :] - projections) * scale_proj #torch.zeros(original_resid_stream[:, position, head, :].shape)#

  return original_resid_stream


# get ldds when intervening and replacing with directions of corrupted runs
def project_away_component_and_replace_with_something_else(
    original_resid_out: Float[Tensor, "batch seq head_idx d_model"],
    hook: HookPoint,
    project_away_vector: Float[Tensor, "batch d_model"],
    replace_vector : Float[Tensor, "batch d_model"],
    position = -1,
    heads = [], # array of ints,
    project_only = False # whether to, instead of projecting away the vector, keep it!
) -> Float[Tensor, "batch n_head pos pos"]:
    '''
    Function which gets removes a specific component (or keeps only it, if project_only = True) of the an output of a head and replaces it with another vector
    project_only: if true, then instead of projecting away the vector, it keeps only it
    '''
    # right now this projects away the IO direction!
    assert project_away_vector.shape == replace_vector.shape and len(project_away_vector.shape) == 2

    for head in heads:
        head_output = original_resid_out[:, position, head, :]
        projections = get_projection(head_output, project_away_vector)

        if project_only:
            resid_without_projection =  projections
        else:
            resid_without_projection = (head_output - projections)

        updated_resid = resid_without_projection + replace_vector
        original_resid_out[:, position, head, :] = updated_resid

    return original_resid_out

def patch_last_ln(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  #print("froze lnfinal")
  ln_scale = clean_cache["ln_final.hook_scale"]
  return ln_scale

unembed_io_directions = model.tokens_to_residual_directions(answer_tokens[:, 0])
#unembed_s_directions = model.tokens_to_residual_directions(answer_tokens[:, 1])
#unembed_diff_directions = unembed_io_directions - unembed_s_directions

target_intervene_direction = unembed_io_directions
ln_on = True
def patch_head_vector(
    head_vector: Float[Tensor, "batch pos head_index d_head"],
    hook: HookPoint,
    head_indices: int,
    other_cache: ActivationCache
) -> Float[Tensor, "batch pos head_index d_head"]:
    '''
    Patches the output of a given head (before it's added to the residual stream) at
    every sequence position, using the value from the other cache.
    '''
    for head_index in head_indices:
      head_vector[:, :, head_index] = other_cache[hook.name][:, :, head_index].clone()
    return head_vector

def patch_ln_scale(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  ln_scale = clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"].clone()
  return ln_scale

def patch_ln2_scale(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  ln_scale = clean_cache["blocks." + str(hook.layer()) + ".ln2.hook_scale"].clone()
  return ln_scale

# def kq_rewrite_hook(
#     internal_value: Float[Tensor, "batch seq head d_head"],
#     hook: HookPoint,
#     head,
#     unnormalized_resid:  Float[Tensor, "batch seq d_model"],
#     vector,
#     act_name,
#     scale = 1,
#     position = -1,
#     pre_ln = True
# ):
#   """
#   replaces keys or queries with a new result which we get from adding a vector to a position at the residual stream
#   head: tuple for head to rewrite keys for
#   unnormalized_resid: stored unnormalized residual stream needed to recalculated activations
#   """

#   print("intervening in query/key")
#   ln1 = model.blocks[hook.layer()].ln1
#   temp_resid = unnormalized_resid.clone()

#   if pre_ln:
#     temp_resid[:, position, :] = temp_resid[:, position, :] + scale * vector
#     normalized_resid = ln1(temp_resid)
#   else:
#     temp_resid = ln1(temp_resid)
#     temp_resid[:, position, :] = temp_resid[:, position, :] + scale * vector
#     normalized_resid = temp_resid


#   assert act_name == "q" or act_name == "k"
#   if act_name == "q":
#     W_Q, b_Q = model.W_Q[head[0], head[1]], model.b_Q[head[0], head[1]]
#     internal_value[..., head[1], :] = einops.einsum(normalized_resid, W_Q, "batch seq d_model, d_model d_head -> batch seq d_head") + b_Q

#   elif act_name == "k":
#     W_K, b_K = model.W_K[head[0], head[1]], model.b_K[head[0], head[1]]
#     internal_value[..., head[1], :] = einops.einsum(normalized_resid, W_K, "batch seq d_model, d_model d_head -> batch seq d_head") + b_K

def project_stuff_on_heads(project_heads, project_only = False, scale_proj = 1, output = "display_logits", freeze_ln = False, return_just_lds = False):

    model.reset_hooks()
    # project_heads is a list of tuples (layer, head). for each layer, write a hook which projects all the heads from the layer
    for layer in range(model.cfg.n_layers):
        key_heads = [head[1] for head in project_heads if head[0] == layer]
        if len(key_heads) > 0:
            #print(key_heads)
            model.add_hook(utils.get_act_name("result", layer), partial(project_vector_operation, vector = target_intervene_direction, heads = key_heads, scale_proj = scale_proj, project_only = project_only))

    if freeze_ln:
        for layer in [9,10,11]:
            model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
            model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)
        model.add_hook("ln_final.hook_scale", patch_last_ln)

    hooked_logits, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    if output == "display_logits":
        return display_all_logits(hooked_cache, comparison=True, logits = hooked_logits, title = f"Projecting {('only' if project_only else 'away')} IO direction in heads {project_heads}", include_incorrect=INCLUDE_INCORRECT)
    elif output == "get_ldd":
        a = calc_all_logit_contibutions(hooked_cache, include_incorrect=INCLUDE_INCORRECT)
        ca = calc_all_logit_contibutions(clean_cache, include_incorrect=INCLUDE_INCORRECT)
        if return_just_lds:
          return a
        else:
          return a - ca
        
  
# %%
def freeze_ln_from_layer(first_layer):
    for layer in range(first_layer, model.cfg.n_layers):
                model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
                model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)

    model.add_hook("ln_final.hook_scale", patch_last_ln)


def run_interventions(return_just_lds = False):
    """
    return_just_lds: whether or not to return logit contributions/diffs (if True), or if to return logit diff diffs (if False)
    """
    target_heads = [(9,6), (9,9)]#, (10,0)]

    
    zero_ablate_all_heads_ldds = project_stuff_on_heads(target_heads, project_only = True, scale_proj = 0, output = "get_ldd", freeze_ln=ln_on, return_just_lds = return_just_lds)
    project_only_io_direction = project_stuff_on_heads(target_heads, project_only = True, scale_proj = 1, output = "get_ldd", freeze_ln=ln_on, return_just_lds = return_just_lds)
    project_away_io_direction = project_stuff_on_heads(target_heads, project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=ln_on, return_just_lds = return_just_lds)

    model.reset_hooks()

    if ln_on: freeze_ln_from_layer(9)

    for head in target_heads:

        # get the output of head on CORRUPTED RUN
        W_O_temp = model.W_O[head[0], head[1]]
        layer_z = corrupted_cache[utils.get_act_name("z", head[0])].clone()
        layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
        output_head = layer_result[:, -1, head[1], :]

        # get projection of CORRUPTED HEAD OUTPUT onto IO token
        corrupted_head_only_IO_output = get_projection(output_head, target_intervene_direction)

        # add hook to now replace with this corrupted IO direction
        model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = target_intervene_direction, heads = [head[1]], replace_vector = corrupted_head_only_IO_output))

    _, replace_with_new_IO_cache = model.run_with_cache(clean_tokens)

    model.reset_hooks()

    if ln_on: freeze_ln_from_layer(9)

    for head in target_heads:
        print(head)
        # get the output of head on CORRUPTED RUN
        W_O_temp = model.W_O[head[0], head[1]]
        layer_z = corrupted_cache[utils.get_act_name("z", head[0])].clone()
        layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
        output_head = layer_result[:, -1, head[1], :]


        # get projection of CORRUPTED HEAD OUTPUT onto IO perp token
        corrupted_head_only_IO_output = get_projection(output_head, target_intervene_direction)
        everything_else_but_that = output_head - corrupted_head_only_IO_output

        # add hook to now replace with this corrupted IO perp direction
        model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = target_intervene_direction, heads = [head[1]], replace_vector = everything_else_but_that, project_only = True))

    _, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()

    print("-----")
    ca = calc_all_logit_contibutions(clean_cache, include_incorrect=INCLUDE_INCORRECT)

    if return_just_lds:
      replace_all_IOs_ldds = calc_all_logit_contibutions(replace_with_new_IO_cache, include_incorrect=INCLUDE_INCORRECT)
      replace_all_perp_IOs_ldds = calc_all_logit_contibutions(replace_with_new_perp_IO_cache, include_incorrect=INCLUDE_INCORRECT)
      return [zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds, project_away_io_direction, replace_all_IOs_ldds]
    else:
      replace_all_IOs_ldds = calc_all_logit_contibutions(replace_with_new_IO_cache, include_incorrect=INCLUDE_INCORRECT)  - ca
      replace_all_perp_IOs_ldds = calc_all_logit_contibutions(replace_with_new_perp_IO_cache, include_incorrect=INCLUDE_INCORRECT)  - ca
      return [zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds, project_away_io_direction, replace_all_IOs_ldds]

# %%
third_intervention = run_interventions(return_just_lds = True)
zero_ablate_all_heads_lds, project_only_io_direction_lds, replace_all_perp_IOs_lds, project_away_io_direction, replace_all_IOs_lds = third_intervention

#per_head_logit_diff, ablated_logit_diff
# %% Plot Interventions on ldd y=x graph
fig = px.scatter()
x =  per_head_logit_diff.flatten()

heads_to_name =  [(10,0), (10,7), (9,9), (9,6), (11,10), (11,2)]
fig_names = [str((i,j)) for i in range(12) for j in range(12)]
for i in range(12):
  for j in range(12):
    if fig_names[i * 12 + j] not in [str(i) for i in heads_to_name]:
      fig_names[i * 12 + j] = None
    else:
      fig_names[i * 12 + j] = str(i) + "." + str(j)



# left

fig.add_trace(go.Scatter(x = x.cpu(), y = zero_ablate_all_heads_lds.flatten().cpu(), text = fig_names ,textposition="top center", mode = 'markers+text', name = "Zero Ablate Directions"))
fig.add_trace(go.Scatter(x = x.cpu(), y = replace_all_perp_IOs_lds.flatten().cpu(),  text = fig_names ,textposition="top center", mode = 'markers+text', name = "Replace All IO-Perp Directions"))
fig.add_trace(go.Scatter(x = x.cpu(), y = project_only_io_direction_lds.flatten().cpu(), text = fig_names , textposition="top center", mode = 'markers+text', name = "Project Only IO Directions"))
fig.add_trace(go.Scatter(x = x.cpu(), y = project_away_io_direction.flatten().cpu(),  text = fig_names ,textposition="top center", mode = 'markers+text', name = "Project Away IO Directions"))
fig.add_trace(go.Scatter(x = x.cpu(), y = replace_all_IOs_lds.flatten().cpu(), text = fig_names , textposition="top center", mode = 'markers+text', name = "Replace All IO Directions"))

x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
# right


# on both
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ))
  #y =  ablated_logit_diff.flatten()
  #fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(),  textposition="top center", mode = 'markers+text', name = "sample ablated", marker=dict(color="purple")), row = 1, col = col)



fig.update_xaxes(title = "Clean Direct Effect")
fig.update_yaxes(title = "Ablated Direct Effect")
fig.update_layout(title = "Logit Differences When Zero Ablating in Name Mover Heads", width = 950)
fig.show()
# %%
# 
def get_average_and_not_output_across_tasks(cache, layer, head) -> Float[Tensor, "pos d_model"]:
    head_outs = cache[utils.get_act_name("z", layer)][..., head, :].clone()
    return head_outs.mean(0).to(device), head_outs.to(device)
    

# %% Major Idea. Mean Ablate all of these heads, and see if backup occurs. I don't know how I haven't tried this yet.
model.reset_hooks()
if ln_on:  freeze_ln_from_layer(9)
for head in [(9,6), (9,9)]:
    # get the average output of head
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z, clean_outputs = get_average_and_not_output_across_tasks(clean_cache, head[0], head[1])
    head_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, seq d_head ->  seq d_model")[-1]
    head_result = einops.repeat(head_result, "d_model -> batch d_model", batch = clean_tokens.shape[0])
    clean_outputs = clean_outputs[:, -1, :]
    clean_outputs = einops.einsum(W_O_temp, clean_outputs, "d_head d_model, seq d_head ->  seq d_model")

    # add hook to now replace with this corrupted IO perp direction
    model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = clean_outputs, heads = [head[1]], replace_vector = head_result, project_only = False))

replace_with_new_perp_IO_logits, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)
model.reset_hooks()
replace_all_perp_IOs_logit_contributions = calc_all_logit_contibutions(replace_with_new_perp_IO_cache)
display_all_logits(replace_with_new_perp_IO_cache, comparison = True, title = f"Logit Diff Diff from Mean Ablating in heads 9.6 and 9.9", include_incorrect=INCLUDE_INCORRECT)
# %% Replace 9.9 and 9.6 with Output - Mean
model.reset_hooks()
if ln_on:  freeze_ln_from_layer(9)
for head in [(9,6), (9,9)]:
    # get the average output of head
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z, clean_outputs = get_average_and_not_output_across_tasks(clean_cache, head[0], head[1])
    head_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, seq d_head ->  seq d_model")[-1]
    head_result = einops.repeat(head_result, "d_model -> batch d_model", batch = clean_tokens.shape[0])
    clean_outputs = clean_outputs[:, -1, :]
    clean_outputs = einops.einsum(W_O_temp, clean_outputs, "d_head d_model, seq d_head ->  seq d_model")

    # add hook to now replace with this corrupted IO perp direction
    model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = clean_outputs, heads = [head[1]], replace_vector = clean_outputs - head_result, project_only = False))

replace_with_new_perp_IO_logits, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)
model.reset_hooks()
replace_all_perp_IOs_logit_contributions = calc_all_logit_contibutions(replace_with_new_perp_IO_cache)
display_all_logits(replace_with_new_perp_IO_cache, comparison = True, title = f"Logit Diff Diff from Clean - Mean in heads 9.6 and 9.9", include_incorrect=INCLUDE_INCORRECT)

# %% Try adding back in the IO directions into the sample ablated stuff?
model.reset_hooks()
if ln_on:  freeze_ln_from_layer(9)

for head in [(9,6), (9,9)]:
    # get the average output of head
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z, clean_outputs = get_average_and_not_output_across_tasks(clean_cache, head[0], head[1])
    
    head_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, seq d_head ->  seq d_model")[-1]
    head_result = einops.repeat(head_result, "d_model -> batch d_model", batch = clean_tokens.shape[0])


    clean_outputs = clean_outputs[:, -1, :]
    clean_outputs = einops.einsum(W_O_temp, clean_outputs, "d_head d_model, seq d_head ->  seq d_model")

    # get projection of CORRUPTED HEAD OUTPUT onto IO perp token
    # corrupted_head_only_IO_output = get_projection(output_head, target_intervene_direction)
    # everything_else_but_that = output_head - corrupted_head_only_IO_output

    # add hook to now replace with this corrupted IO perp direction
    print(clean_outputs.shape)
    print(head_result.shape)
    model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = clean_outputs, heads = [head[1]], replace_vector = head_result + 4 * unembed_io_directions, project_only = False))

replace_with_new_perp_IO_logits, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)
model.reset_hooks()

ca = calc_all_logit_contibutions(clean_cache)
replace_all_perp_IOs_logit_contributions = calc_all_logit_contibutions(replace_with_new_perp_IO_cache)
display_all_logits(replace_with_new_perp_IO_cache, comparison = True, title = f"Logit Diff Diff from Mean Ablating, plus IO unembeddings, in heads 9.6 and 9.9", include_incorrect=INCLUDE_INCORRECT)
# %% it appears that adding the IO direction causes the backup heads to change the amount of backup they show. lets actually test this out further

io_scaling = [0.1 * i for i in range(-70,70,1)]
heads_to_graph = [(10,7), (10,2)]
head_changed_outputs_to_io_direction = torch.zeros((len(heads_to_graph), len(io_scaling)))

# generate a random vector that is the same size as unembed_io_directions
random_vector = torch.randn(unembed_io_directions.shape).to(device) * 10
# make it the same length as unembed_io_direction
random_vector = random_vector / torch.norm(random_vector, dim = -1, keepdim = True) * torch.norm(unembed_io_directions, dim = -1, keepdim = True)

# run interventions
for scaling in io_scaling:
    model.reset_hooks()
    if ln_on:
        for layer in [9,10,11]:
                    model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
                    model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)

    model.add_hook("ln_final.hook_scale", patch_last_ln)
    for head in [(9,6), (9,9)]:
        # get the average output of head
        W_O_temp = model.W_O[head[0], head[1]]
        layer_z, clean_outputs = get_average_and_not_output_across_tasks(clean_cache, head[0], head[1])
        
        head_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, seq d_head ->  seq d_model")[-1]
        head_result = einops.repeat(head_result, "d_model -> batch d_model", batch = clean_tokens.shape[0])


        clean_outputs = clean_outputs[:, -1, :]
        clean_outputs = einops.einsum(W_O_temp, clean_outputs, "d_head d_model, seq d_head ->  seq d_model")

        # get projection of CORRUPTED HEAD OUTPUT onto IO perp token
        # corrupted_head_only_IO_output = get_projection(output_head, target_intervene_direction)
        # everything_else_but_that = output_head - corrupted_head_only_IO_output

        # add hook to now replace with this corrupted IO perp direction
    
        
        model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = clean_outputs, heads = [head[1]], replace_vector = clean_outputs + scaling * unembed_io_directions, project_only = False))

    replace_with_new_perp_IO_logits, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()

    ca = calc_all_logit_contibutions(clean_cache, per_prompt=True, include_incorrect=INCLUDE_INCORRECT)
    replace_all_perp_IOs_logit_contributions = calc_all_logit_contibutions(replace_with_new_perp_IO_cache, per_prompt=True, include_incorrect=INCLUDE_INCORRECT)

    for index, head in enumerate(heads_to_graph):
        head_changed_outputs_to_io_direction[index, io_scaling.index(scaling)] = replace_all_perp_IOs_logit_contributions[head[0], head[1]].mean(0) - ca[head[0], head[1]].mean(0)

fig = go.Figure()
x = io_scaling

for head in heads_to_graph:
    fig.add_trace(go.Scatter(x = x, y = head_changed_outputs_to_io_direction[heads_to_graph.index(head)], mode = 'lines', name = f"Head {head}"))

fig.update_xaxes(title = "Scaling of Direction")
fig.update_yaxes(title = "Logit Contribution Difference")
fig.update_layout(
    title = "Logit Contribution Difference when adding the io direction to the output of 9.6 and 9.9"
)
fig.show()


# %% Replace the residual stream in layer 11 with layer 10, and see change in logit diff

def replace_resid_stream_hook(resid_stream, hook, resid_stream_to_replace_with):
    print("changing residual stream")
    resid_stream = resid_stream_to_replace_with.clone()
    return resid_stream

clone_layer = 10
into_layer = 11
model.reset_hooks()
resid_pre_10 = clean_cache[utils.get_act_name("resid_pre", clone_layer)].clone()
model.add_hook(utils.get_act_name("resid_pre", into_layer), partial(replace_resid_stream_hook, resid_stream_to_replace_with = resid_pre_10))
_, hooked_cache = model.run_with_cache(clean_tokens)
model.reset_hooks()
# display new logit diff
display_all_logits(hooked_cache, comparison = True, title = f"Logit Diff Diff from cloning resid_pre_{clone_layer} into resid_pre_{into_layer}", include_incorrect=INCLUDE_INCORRECT)
del resid_pre_10

# %%
def kqv_rewrite_hook(
    internal_value: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    head: tuple,
    changed_resid:  Float[Tensor, "batch seq d_model"],
    act_name,
    position = -1,
    pre_ln = True
):
  """
  replaces keys or queries with a new result which we get from a different residual stream
  head: tuple for head to rewrite keys for
  changed_resid: stored unnormalized residual stream needed to recalculated activations
  """

  #print("intervening in query/key")

  #assert changed_resid.equal(clean_cache[utils.get_act_name("resid_pre", 9)])
  ln1 = model.blocks[head[0]].ln1
  #print(ln1)
  temp_resid = changed_resid.clone()
  normalized_resid = ln1(temp_resid)

  assert act_name == "q" or act_name == "k" or act_name == "v"
  if act_name == "q":
    weight, bias = (model.W_Q[head[0], head[1]], model.b_Q[head[0], head[1]])
  elif act_name == "k":
    weight, bias = model.W_K[head[0], head[1]], model.b_K[head[0], head[1]]
  else:
    weight, bias =  model.W_V[head[0], head[1]], model.b_V[head[0], head[1]]

  temp = einops.einsum(normalized_resid, weight, "batch seq d_model, d_model d_head -> batch seq d_head") + bias
#   print(temp[0][0])
#   print(internal_value[..., head[1], :][0][0])
#   assert internal_value[..., head[1], :].equal(temp)
  internal_value[..., head[1], :] = temp
# %% Add the new residual stream into components of a head
clone_layer = 9
into_layer = 11
heads_to_write_into = [(into_layer,j) for j in range(12)] #+  [(11,i) for i in range(12)]

for act, act_name in [("q", "query"), ("k", "key"),  ("v", "value")]:
    model.reset_hooks()
    resid_pre = clean_cache[utils.get_act_name("resid_mid", clone_layer)].clone()
    for head in heads_to_write_into:
        model.add_hook(utils.get_act_name(act, head[0]), partial(kqv_rewrite_hook, changed_resid = resid_pre, head = head, act_name = act))
        pass
    _, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    # display new logit diff
    display_all_logits(hooked_cache, comparison = True, title = f"Logit Diff Diff from cloning resid_pre_{clone_layer} into the {act_name} of {heads_to_write_into}", include_incorrect=INCLUDE_INCORRECT)


# %% Add the new residual stream + an output of a head into components of a head
clone_layer = 9
into_layer = 10
heads_to_write_into = [(into_layer,j) for j in range(12)] #+  [(11,i) for i in range(12)]

#heads_whose_outputs_to_simulate = [(9,6), (9,9)]
heads_whose_outputs_to_simulate =  []#[(9,i) for i in range(12) if (9,i) not in []]
resid_pre = clean_cache[utils.get_act_name("resid_mid", clone_layer)].clone()
temp_checker = torch.zeros(resid_pre.shape).cuda()
# add head outputs to resid_pre
for head in heads_whose_outputs_to_simulate:
    #resid_pre += clean_cache[utils.get_act_name("result", head[0])][:, :, head[1], :]
    temp_checker += clean_cache[utils.get_act_name("result", head[0])][:, :, head[1], :].clone()
if False:
    #resid_pre += model.b_O[clone_layer]
    temp_checker += model.b_O[clone_layer]

resid_pre += temp_checker


for act, act_name in [("q", "query"), ("k", "key"),  ("v", "value")]:
    model.reset_hooks()
    for head in heads_to_write_into:
        model.add_hook(utils.get_act_name(act, head[0]), partial(kqv_rewrite_hook, changed_resid = resid_pre, head = head, act_name = act))
    _, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    # display new logit diff
    display_all_logits(hooked_cache, comparison = True, title = f"Logit Diff Diff from cloning resid_pre_{clone_layer} into the {act_name} of downstream layers, while simulating output of {heads_whose_outputs_to_simulate}", include_incorrect=INCLUDE_INCORRECT)

# %%
print(clean_cache[utils.get_act_name("resid_mid", 9)][0][0][0:30])
print(clean_cache[utils.get_act_name("resid_pre", 9)][0][0][0:30] + clean_cache[utils.get_act_name("attn_out", 9)][0][0][0:30])


assert clean_cache[utils.get_act_name("resid_mid", 9)].allclose(clean_cache[utils.get_act_name("resid_pre", 9)] + clean_cache[utils.get_act_name("attn_out", 9)])
# %%
