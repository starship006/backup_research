# %% [markdown]
# # Imports

# %%
# %%
%pip install git+https://github.com/neelnanda-io/TransformerLens.git
%pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
%pip install plotly

# %%

from imports import *

# %%
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,

    device = device
)
# %% 
model.use_attn_result = True

# %%
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

# %% [markdown]
# ## code
# 

# %%
template = "When{name_A} and{name_B} went to the {place},{name_C} gave the {object} to"

# %%
# prompt: generate a list of prompts using the template. ensure that no prompt uses the same two names

import random

# Create a list of all possible pairs of names
names_list = list(itertools.combinations(names, 2))

# Create a list of prompts
prompts = []
counter_prompts = []


name_pairs = []
#counter_name_pairs = []

for name_pair in names_list:
    name_A, name_B = name_pair
    # Generate a random place
    place = random.choice(places)
    # Generate a random object
    objectA = random.choice(objects)
    # Create a prompt
    prompt = template.format(
        name_A=name_A,
        name_B=name_B,
        place=place,
        name_C=name_B,
        object=objectA,
    )
    prompts.append(prompt)
    name_pairs.append([name_A, name_B])

    # generate flipped
    prompt = template.format(
        name_A=name_B,
        name_B=name_A,
        place=place,
        name_C=name_B,
        object=objectA,
    )
    prompts.append(prompt)
    name_pairs.append([name_A, name_B])



    # generate three other names that are not name_A and name_B
    other_names = []
    while(len(other_names) != 3):
      new_name = random.choice(names)
      if new_name is not name_A and new_name is not name_B and new_name not in other_names:
        other_names.append(new_name)

    counter_prompts.append(template.format(

        name_A=other_names[0],
        name_B=other_names[1],
        place=place,
        name_C=other_names[2],
        object=objectA,

    ))

    counter_prompts.append(template.format(

        name_A=other_names[1],
        name_B=other_names[0],
        place=place,
        name_C=other_names[2],
        object=objectA,

    ))

# Print the prompts

# %%
# generate random list of numbers
results = []
for i in tqdm(range(1000)):

    rand_indices = torch.randint(0, len(prompts), size = (350,))

    # indices plus the indices + 1
    #double_rand_indices = torch.cat((rand_indices, rand_indices + 1))


    rand_indices

   
    clean_prompts = [prompts[i] for i in rand_indices]
    corrupted_prompts = [counter_prompts[i] for i in rand_indices]
    name_answers = [name_pairs[i] for i in rand_indices]


   
    clean_tokens = model.to_tokens(clean_prompts, prepend_bos = True).cuda()
    corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos = True).cuda()
    answer_tokens = torch.concat([
        model.to_tokens(names, prepend_bos=False).squeeze(dim=1).unsqueeze(dim=0) for names in name_answers
    ]).cuda()

   
    clean_tokens.shape

   
    index = 1
    clean_prompts[index], corrupted_prompts[index], name_answers[index]

   
    model.reset_hooks()
    clean_logits, clean_cache = model.run_with_cache(clean_tokens, prepend_bos = False)
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens, prepend_bos = False)

   
    torch.cuda.empty_cache()

    def logits_to_ave_logit_diff(
        logits: Float[Tensor, "batch seq d_vocab"],
        answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
        per_prompt: bool = False
    ):
        '''
        Returns logit difference between the correct and incorrect answer.

        If per_prompt=True, return the array of differences rather than the average.
        '''
        # Only the final logits are relevant for the answer
        final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
        # Get the logits corresponding to the indirect object / subject tokens respectively
        answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens)
        # Find logit difference
        correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
        answer_logit_diff = correct_logits - incorrect_logits
        return answer_logit_diff if per_prompt else answer_logit_diff.mean()

   
    clean_per_prompt_diff = logits_to_ave_logit_diff(clean_logits, per_prompt = True)

    clean_average_logit_diff = logits_to_ave_logit_diff(clean_logits)
    corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits)

    # print(clean_average_logit_diff)
    # print(corrupted_average_logit_diff)

   
    answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
    correct_residual_direction, incorrect_residual_direction = answer_residual_directions.unbind(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_direction - incorrect_residual_direction

   
    def residual_stack_to_logit_diff(
        residual_stack: Float[Tensor, "... batch d_model"],
        cache: ActivationCache,
        clean_cache = clean_cache,
        logit_diff_directions: Float[Tensor, "batch d_model"] = logit_diff_directions,
        use_clean_cache_for_LN = True
    ) -> Float[Tensor, "..."]:
        '''
        Gets the avg logit difference between the correct and incorrect answer for a given
        stack of components in the residual stream.
        '''




        batch_size = residual_stack.size(-2)
        if use_clean_cache_for_LN:
            scaled_residual_stack = clean_cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)
        else:
            scaled_residual_stack = cache.apply_ln_to_stack(residual_stack, layer=-1, pos_slice=-1)



        # # some extra code for more sanity checking
        # new_logits = scaled_residual_stack @ model.W_U
        # print(new_logits.shape)
        # new_logits = einops.repeat(new_logits, "batch d_vocab -> batch 1 d_vocab")
        # print(new_logits.shape)
        # print(logits_to_ave_logit_diff(new_logits))

        return einops.einsum(
            scaled_residual_stack, logit_diff_directions,
            "... batch d_model, batch d_model -> ..."
        ) / batch_size


   
    answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
    #print("Answer residual directions shape:", answer_residual_directions.shape)

    correct_residual_directions, incorrect_residual_directions = answer_residual_directions.unbind(dim=1)
    logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_directions - incorrect_residual_directions
    #print(f"Logit difference directions shape:", logit_diff_directions.shape)

   
    model.b_U.shape

   
    diff_from_unembedding_bias = model.b_U[answer_tokens[:, 0]] -  model.b_U[answer_tokens[:, 1]]

   
    final_residual_stream: Float[Tensor, "batch seq d_model"] = clean_cache["resid_post", -1]
    #print(f"Final residual stream shape: {final_residual_stream.shape}")
    final_token_residual_stream: Float[Tensor, "batch d_model"] = final_residual_stream[:, -1, :]

    #print(f"Calculated average logit diff: {(residual_stack_to_logit_diff(final_token_residual_stream, clean_cache, logit_diff_directions = logit_diff_directions) + diff_from_unembedding_bias.mean(0)):.10f}") # <-- okay b_U exists... and matters
    #print(f"Original logit difference:     {clean_average_logit_diff:.10f}")
    # ## Logit Diffs + Gather Important Heads

    def calc_all_logit_diffs(cache, use_clean_cache = True):
        clean_per_head_residual, labels = cache.stack_head_results(layer = -1, return_labels = True, apply_ln = False) # per_head_residual.shape = heads batch seq_pos d_model
        # also, for the worried, no, we're not missing the application of LN here since it gets applied in the below function call
        per_head_logit_diff: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, use_clean_cache_for_LN=use_clean_cache)

        per_head_logit_diff = einops.rearrange(
            per_head_logit_diff,
            "(layer head) ... -> layer head ...",
            layer=model.cfg.n_layers
        )

        correct_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = correct_residual_direction, use_clean_cache_for_LN=use_clean_cache)

        correct_direction_per_head_logit = einops.rearrange(
            correct_direction_per_head_logit,
            "(layer head) ... -> layer head ...",
            layer=model.cfg.n_layers
        )

        incorrect_direction_per_head_logit: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], cache, logit_diff_directions = incorrect_residual_direction, use_clean_cache_for_LN=use_clean_cache)

        incorrect_direction_per_head_logit = einops.rearrange(
            incorrect_direction_per_head_logit,
            "(layer head) ... -> layer head ...",
            layer=model.cfg.n_layers
        )

        return per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit

    per_head_logit_diff, correct_direction_per_head_logit, incorrect_direction_per_head_logit = calc_all_logit_diffs(clean_cache)

   
    top_heads = []
    k = 5

    flattened_tensor = per_head_logit_diff.flatten().cpu()
    _, topk_indices = torch.topk(flattened_tensor, k)
    top_layer_arr, top_index_arr = np.unravel_index(topk_indices.numpy(), per_head_logit_diff.shape)

    for l, i in zip(top_layer_arr, top_index_arr):
        top_heads.append((l,i))

    #print(top_heads)

   
    per_head_logit_diff[11]

   
    neg_heads = []
    neg_indices = torch.nonzero(torch.lt(per_head_logit_diff, -0.1))
    neg_heads_list = neg_indices.squeeze().tolist()
    for i in neg_heads_list:
        neg_heads.append((i[0], i[1]))

    #print(neg_heads)

   
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
        else:
            fig.show()
    fig = display_all_logits(clean_cache, title = "Logit Contributions on Clean Dataset", return_fig = True, logits = clean_logits)
    def stare_at_attention_and_head_pat(cache, layer_to_stare_at, head_to_isolate, display_corrupted_text = False, verbose = True, specific = False, specific_index = 0):
        """
        given a cache from a run, displays the attention patterns of a layer, as well as printing out how much the model
        attends to the S1, S2, and IO token
        """

        tokenized_str_tokens = model.to_str_tokens(corrupted_tokens[0]) if display_corrupted_text else model.to_str_tokens(clean_tokens[0])
        attention_patten = cache["pattern", layer_to_stare_at]
        print(f"Layer {layer_to_stare_at} Head {head_to_isolate} Activation Patterns:")


        if not specific:
            S1 = attention_patten.mean(0)[head_to_isolate][-1][2].item()
            IO = attention_patten.mean(0)[head_to_isolate][-1][4].item()
            S2 = attention_patten.mean(0)[head_to_isolate][-1][10].item()
        else:
            S1 = attention_patten[specific_index, head_to_isolate][-1][2].item()
            IO = attention_patten[specific_index, head_to_isolate][-1][4].item()
            S2 = attention_patten[specific_index, head_to_isolate][-1][10].item()


        print("Attention on S1: " + str(S1))
        print("Attention on IO: " + str(IO))
        print("Attention on S2: " + str(S2))
        print("S1 + IO - S2 = " + str(S1 + IO - S2))
        print("S1 + S2 - IO = " + str(S1 + S2 - IO))
        print("S1 - IO - S2 = " + str(S1 - S2 - IO))


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

    # def display_corrupted_clean_logits(cache, title = "Logit Contributions", comparison = False, return_fig = False, logits = None):

    #   a,b,c = calc_all_logit_diffs(cache)
    #   if logits is not None:
    #     ld = logits_to_ave_logit_diff(logits)
    #   else:
    #     ld = 0.00

    #   if not comparison:
    #     fig = imshow(
    #         torch.stack([a]),
    #         return_fig = True,
    #         facet_col = 0,
    #         facet_labels = [f"Logit Diff - {ld:.2f}"],
    #         title=title,
    #         labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
    #         #coloraxis=dict(colorbar_ticksuffix = "%"),
    #         border=True,
    #         width=1500,
    #         margin={"r": 100, "l": 100}
    #     )
    #   else:

    #     ca, cb, cc = calc_all_logit_diffs(clean_cache)
    #     fig = imshow(
    #         torch.stack([a, ca, a - ca]),
    #         return_fig = True,
    #         facet_col = 0,
    #         facet_labels = [f"Ablated Logit Differences: {ld:.2f}", f"Clean Logit Differences: {clean_average_logit_diff:.2f}", f"Difference between Ablated and Clea: {(ld - clean_average_logit_diff):.2f}",],
    #         title=title,
    #         labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
    #         #coloraxis=dict(colorbar_ticksuffix = "%"),
    #         border=True,
    #         width=1000,
    #         margin={"r": 100, "l": 100}
    #     )


    #   if return_fig:
    #     return fig
    #   else:
    #     fig.show()
        
    #     return a - ca

    # heads =  [(9,9), (9,6), (10,0)]
    # model.reset_hooks() # callum library buggy
    # def return_item(item):
    #   return item

    # model.reset_hooks()
    # patched_logits = act_patch(
    #     model = model,
    #     orig_input = clean_tokens,
    #     new_cache = corrupted_cache,
    #     patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    #     patching_metric = return_item,
    #     verbose = False,
    #     apply_metric_to_cache = False
    # )

    # model.reset_hooks()
    # noise_sample_ablating_results = act_patch(
    #     model = model,
    #     orig_input = clean_tokens,
    #     new_cache = corrupted_cache,
    #     patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    #     patching_metric = partial(display_corrupted_clean_logits, title = f"Logit Differences When Sample Ablating in Name Mover Heads", comparison = True, logits = patched_logits),
    #     verbose = False,
    #     apply_metric_to_cache = True
    # )

    # # #  Graph Figure

    #
    # heads =  [(9,i) for i in range(12)] +  [(10,i) for i in range(12)]
    # model.reset_hooks() # callum library buggy
    def return_item(item):
      return item

    # model.reset_hooks()
    # patched_logits = act_patch(
    #     model = model,
    #     orig_input = clean_tokens,
    #     new_cache = corrupted_cache,
    #     patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    #     patching_metric = return_item,
    #     verbose = False,
    #     apply_metric_to_cache = False
    # )

    # model.reset_hooks()
    # all_layernine_noise_patching_results = act_patch(
    #     model = model,
    #     orig_input = clean_tokens,
    #     new_cache = corrupted_cache,
    #     patching_nodes = [Node("z", layer = layer, head = head) for layer, head in heads],
    #     patching_metric = partial(display_corrupted_clean_logits, title = f"Logits When Sample Ablating in Layer 9", comparison = True, logits = patched_logits),
    #     verbose = False,
    #     apply_metric_to_cache = True
    # )

    #
    # layer_nine_patching_NMHs = act_patch(
    #     model = model,
    #     orig_input = clean_tokens,
    #     new_cache = corrupted_cache,
    #     patching_nodes = [Node("z", layer = 9 , head = 6) , Node("z", layer = 9 , head = 9)],
    #     patching_metric = return_item,
    #     verbose = False,
    #     apply_metric_to_cache = True
    # )

    #
    # ablated_logit_diff,_,_ = calc_all_logit_diffs(layer_nine_patching_NMHs)

    #
    # neg_m_heads = [(10,7), (11,10)]    
    # name_mover_heads = [(9,9), (9,6), (10,0)]
    # backup_heads = [(9,0), (9,7), (10,1), (10,2), (10,6), (10,10), (11,2), (11,9)]
    # key_backup_heads = [(10,2), (10,6), (10,10), (11,2)]
    # strong_neg_backup_heads = [(11,2), (10,2), (10,0), (11,6)]

    #
    # heads_to_name = neg_m_heads + [(10,0)] + key_backup_heads
    # fig_names = [str((i,j)) for i in range(12) for j in range(12)]
    # for i in range(12):
    #   for j in range(12):
    #     if fig_names[i * 12 + j] not in [str(i) for i in heads_to_name]:
    #       fig_names[i * 12 + j] = None
    #     else:
    #       fig_names[i * 12 + j] = str(i) + "." + str(j)



    #
    # x =  per_head_logit_diff.flatten()
    # y =  ablated_logit_diff.flatten()


    # fig = px.scatter()

    # fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(), text = fig_names, textposition="top center", mode = 'markers+text', name = "gpt-2"))

    # x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
    # fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ))
    # fig.update_xaxes(title = "Clean Logit Difference")
    # fig.update_yaxes(title = "Post-Intervention Logit Difference")
    # fig.update_layout(title = "Logit Differences When Sample Ablating in Layer 9 Name Mover Heads", width = 950)
    # fig.show()


    # # save fig as pdf
    # fig.write_image("ldd_sample_ablation.pdf")


    # ## Figures: How much does Negative Heads explain Self-Repair?

    cache_patching_NMHs = act_patch(
        model = model,
        orig_input = clean_tokens,
        new_cache = corrupted_cache,
        patching_nodes = [Node("z", layer = 9 , head = 6) , Node("z", layer = 9 , head = 9), Node("z", layer = 10 , head = 0)],
        patching_metric = return_item,
        verbose = False,
        apply_metric_to_cache = True
    )

    patched_NMHs_logit_diff ,_,_ = calc_all_logit_diffs(cache_patching_NMHs)
    patched_NMHS_backup = patched_NMHs_logit_diff - per_head_logit_diff
    assert (patched_NMHS_backup[0:10].flatten().sum() - patched_NMHS_backup[9,9] - patched_NMHS_backup[9,6]).isclose(torch.tensor([0.0]).cuda(), atol = 1e-5)
    last_two_layer_diff = patched_NMHS_backup.flatten().sum() - patched_NMHS_backup[10, 0] - patched_NMHS_backup[9,9] - patched_NMHS_backup[9,6]
    sum_from_negative = patched_NMHS_backup[10, 7] + patched_NMHS_backup[11,10]
    neg_head_backup_amount = sum_from_negative / last_two_layer_diff
    #print(neg_head_backup_amount)
    results.append(neg_head_backup_amount)
    #print(last_two_layer_diff)

# %%
# average results
print(sum(results) / len(results))

# %% [markdown]
# # Normal

# %%
neg_m_heads = [(10,7), (11,10)]
name_mover_heads = [(9,9), (9,6), (10,0)]
backup_heads = [(9,0), (9,7), (10,1), (10,2), (10,6), (10,10), (11,2), (11,9)]
key_backup_heads = [(10,2), (10,6), (10,10), (11,2)]
strong_neg_backup_heads = [(11,2), (10,2), (10,0), (11,6)]



head_names = ["Negative", "Name Mover", "Backup"]
head_list = [neg_m_heads, name_mover_heads, backup_heads]

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
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff))

print(f"IOI metric (IOI dataset): {noising_ioi_metric(clean_logits):.4f}")
print(f"IOI metric (ABC dataset): {noising_ioi_metric(corrupted_logits):.4f}")

# %%
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
    return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff) + 1)


print(f"IOI metric (IOI dataset): {denoising_ioi_metric(clean_logits):.4f}")
print(f"IOI metric (ABC dataset): {denoising_ioi_metric(corrupted_logits):.4f}")

# %% [markdown]
# ## Query Intervention

# %%
def store_activation(
    activation,
    hook: HookPoint,
    where_to_store
):
    """
    takes a storage container where_to_store, and stores the activation in it at a hook
    """""
    where_to_store[:] = activation

# %%
def kq_rewrite_hook(
    internal_value: Float[Tensor, "batch seq head d_head"],
    hook: HookPoint,
    head,
    unnormalized_resid:  Float[Tensor, "batch seq d_model"],
    vector,
    act_name,
    scale = 1,
    position = -1,
    pre_ln = True
):
  """
  replaces keys or queries with a new result which we get from adding a vector to a position at the residual stream
  head: tuple for head to rewrite keys for
  unnormalized_resid: stored unnormalized residual stream needed to recalculated activations
  """

  ln1 = model.blocks[hook.layer()].ln1
  temp_resid = unnormalized_resid.clone()

  if pre_ln:
    temp_resid[:, position, :] = temp_resid[:, position, :] + scale * vector
    normalized_resid = ln1(temp_resid)
  else:
    temp_resid = ln1(temp_resid)
    temp_resid[:, position, :] = temp_resid[:, position, :] + scale * vector
    normalized_resid = temp_resid


  assert act_name == "q" or act_name == "k"
  if act_name == "q":
    W_Q, b_Q = model.W_Q[head[0], head[1]], model.b_Q[head[0], head[1]]
    internal_value[..., head[1], :] = einops.einsum(normalized_resid, W_Q, "batch seq d_model, d_model d_head -> batch seq d_head") + b_Q

  elif act_name == "k":
    W_K, b_K = model.W_K[head[0], head[1]], model.b_K[head[0], head[1]]
    internal_value[..., head[1], :] = einops.einsum(normalized_resid, W_K, "batch seq d_model, d_model d_head -> batch seq d_head") + b_K


# %%
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
      head_vector[:, :, head_index] = other_cache[hook.name][:, :, head_index]
    return head_vector

# %%
def patch_ln_scale(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  ln_scale = clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]
  return ln_scale


def patch_ln2_scale(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  ln_scale = clean_cache["blocks." + str(hook.layer()) + ".ln2.hook_scale"]
  return ln_scale

# %%
def causal_write_into_component(act_comp, head, direction, x, pre_ln = True, result_cache_function = None, result_cache_fun_has_head_input = False, freeze_layernorm = False, ablate_heads = []):
  '''
  writes a vector into the component at a given head
  returns new logit differences of run by default, or pass result_cache_funciton to run on cache

  head - tuple for head to intervene in act_comp for
  direction - vector to add to the act_comp in the head
  x - tensor of amount to scale
  '''
  y = torch.zeros(x.shape)
  for i in range(len(x)):
    scale = x[i]
    model.reset_hooks()
    temp = torch.zeros((batch_size, seq_len, model.cfg.d_model)).cuda()
    model.add_hook(utils.get_act_name("resid_pre", head[0]), partial(store_activation, where_to_store = temp))
    if freeze_layernorm:
      model.add_hook("blocks." + str(head[0]) + ".ln1.hook_scale", patch_ln_scale)
    model.add_hook(utils.get_act_name(act_comp, head[0]), partial(kq_rewrite_hook, head = head, unnormalized_resid = temp, vector = direction, act_name = act_comp, scale = scale, pre_ln = pre_ln))


    if len(ablate_heads) != 0:
      for j in ablate_heads:
        model.add_hook(utils.get_act_name("z", j[0]), partial(patch_head_vector, head_indices = [j[1]], other_cache = corrupted_cache))


    hooked_logits, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()


    if result_cache_function != None:
      if not result_cache_fun_has_head_input:
        y[i] = result_cache_function(hooked_cache)
      else:
        y[i] = result_cache_function(hooked_cache, head)
    else:
      # just calculate logit diff
      y[i] = logits_to_ave_logit_diff(hooked_logits)

  return y


# %%
def graph_lines(results, heads, x, title = "Effect of adding/subtracting direction", xtitle = "Scaling on direction", ytitle = "Logit Diff"):
  fig = px.line(title = title)
  for i in range(len(results)):
    fig.add_trace(go.Scatter(x = x, y = results[i], name = str(heads[i])))

  fig.update_xaxes(title = xtitle)
  fig.update_yaxes(title = ytitle)
  fig.show()

# %%
def get_head_IO_minus_S_attn(cache, head, scores = True):

  layer, h_index = head

  if scores:
    attention_patten = cache["attn_scores", layer]
  else:
    attention_patten = cache["pattern", layer]
  S1 = attention_patten.mean(0)[h_index][-1][2].item()
  IO = attention_patten.mean(0)[h_index][-1][4].item()
  S2 = attention_patten.mean(0)[h_index][-1][10].item()

  return IO - S1 - S2


def get_head_IO_minus_just_S1_attn(cache, head, scores = True):

    layer, h_index = head

    if scores:
      attention_patten = cache["attn_scores", layer]
    else:
      attention_patten = cache["pattern", layer]
    S1 = attention_patten.mean(0)[h_index][-1][2].item()
    IO = attention_patten.mean(0)[h_index][-1][4].item()
    S2 = attention_patten.mean(0)[h_index][-1][10].item()

    return IO - S1

def get_head_last_token(cache, head):
  layer, h_index = head
  return cache["pattern", layer][:, h_index, -1, :]


def get_head_attn(cache, head, token, scores = True, mean = True):

  layer, h_index = head

  if scores:
    attention_patten = cache["attn_scores", layer]
  else:
    attention_patten = cache["pattern", layer]


  if mean:
    if token == "S1":
      return attention_patten.mean(0)[h_index][-1][2].item()
    elif token == "IO":
      return attention_patten.mean(0)[h_index][-1][4].item()
    elif token == "S2":
      return attention_patten.mean(0)[h_index][-1][10].item()
    elif token == "BOS":
      return attention_patten.mean(0)[h_index][-1][0].item()
    else:
      print("RAHHHHH YOU MISSTYPED SOMETHING")

  else:
    if token == "S1":
      return attention_patten[:, h_index, -1, 2]
    elif token == "IO":
      return attention_patten[:, h_index, -1, 4]
    elif token == "S2":
      return attention_patten[:, h_index, -1, 10]
    elif token == "BOS":
      return attention_patten[:, h_index, -1, 0]
    else:
      print("RAHHHHH YOU MISSTYPED SOMETHING")


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
      head_vector[:, :, head_index] = other_cache[hook.name][:, :, head_index]
    return head_vector

def get_attn_results_into_head_dirs(heads, direction, scale_amounts, ablate_heads = [], freeze_ln = False, only_S1 = False):
  io_attn_postln_nmh_results = []
  for i in range(len(heads)):
    io_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln, result_cache_function = partial(get_head_attn, token = "IO"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))


  s1_attn_postln_nmh_results = []
  for i in range(len(heads)):
    s1_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = partial(get_head_attn, token = "S1"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))

  s2_attn_postln_nmh_results = []
  for i in range(len(heads)):
    s2_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = partial(get_head_attn, token = "S2"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))

  diff_results = []
  if not only_S1:
    for i in range(len(heads)):
      diff_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                          pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = get_head_IO_minus_S_attn, result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))
  else:
    for i in range(len(heads)):
      diff_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                          pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = get_head_IO_minus_just_S1_attn, result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))


  bos_attn_postln_nmh_results = []
  for i in range(len(heads)):
    bos_attn_postln_nmh_results.append(causal_write_into_component("q", heads[i], direction, scale_amounts,
                                                        pre_ln = True, freeze_layernorm = freeze_ln,result_cache_function = partial(get_head_attn, token = "BOS"), result_cache_fun_has_head_input = True, ablate_heads=ablate_heads))

  return [io_attn_postln_nmh_results, s1_attn_postln_nmh_results, s2_attn_postln_nmh_results, diff_results, bos_attn_postln_nmh_results]

# %%
IO_unembed_direction = model.W_U.T[clean_tokens][:, 4, :]

# %% [markdown]
# # Unembedding to Not Ratios

# %%
model.set_use_attn_result(True)

# %%
def get_projection(from_vector, to_vector):
    dot_product = einops.einsum(from_vector, to_vector, "batch d_model, batch d_model -> batch")
    #print("Average Dot Product of Output Across Batch: " + str(dot_product.mean(0)))
    length_of_from_vector = einops.einsum(from_vector, from_vector, "batch d_model, batch d_model -> batch")
    length_of_vector = einops.einsum(to_vector, to_vector, "batch d_model, batch d_model -> batch")




    projected_lengths = (dot_product) / (length_of_vector)
    #print( einops.repeat(projected_lengths, "batch -> batch d_model", d_model = model.cfg.d_model)[0])
    projections = to_vector * einops.repeat(projected_lengths, "batch -> batch d_model", d_model = to_vector.shape[-1])
    return projections

# %%
a = torch.Tensor([[-1, 1]])
b = torch.Tensor([[1, 1]])
print(get_projection(a, b))

# %%
import torch.nn.functional as F

def compute_cosine_similarity(tensor1, tensor2):
    # Compute cosine similarity
    similarity = F.cosine_similarity(tensor1, tensor2, dim=1)
    return similarity

# %%
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

# %%
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

# %%
def patch_last_ln(ln_scale, hook):
  #print(torch.equal(ln_scale, clean_cache["blocks." + str(hook.layer()) + ".ln1.hook_scale"]))
  print("froze lnfinal")
  ln_scale = clean_cache["ln_final.hook_scale"]
  return ln_scale

# %%
unembed_io_directions = model.tokens_to_residual_directions(answer_tokens[:, 0])
unembed_s_directions = model.tokens_to_residual_directions(answer_tokens[:, 1])
unembed_diff_directions = unembed_io_directions - unembed_s_directions

target_intervene_direction = unembed_io_directions
ln_on = True
ca, cb, cc = calc_all_logit_diffs(clean_cache)

# %%
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
        return display_all_logits(hooked_cache, comparison=True, logits = hooked_logits, title = f"Projecting {('only' if project_only else 'away')} IO direction in heads {project_heads}")
    elif output == "get_ldd":
        a,_,_ = calc_all_logit_diffs(hooked_cache)

        if return_just_lds:
          return a
        else:
          return a - ca

# %%
def compare_intervention_ldds_with_sample_ablated(all_ldds, ldds_names, heads = key_backup_heads, just_logits = False):
    results = torch.zeros((len(all_ldds), len(heads)))


    if just_logits:
        for ldd_index, compare_ldds in enumerate(all_ldds):
            for i, head in enumerate(heads):
                #print(head)
                results[ldd_index, i] = ((compare_ldds[head[0], head[1]]).item()) # / noise_sample_ablating_results[head[0], head[1]]).item())
    else:
        for ldd_index, compare_ldds in enumerate(all_ldds):
            for i, head in enumerate(heads):
                #print(head)
                results[ldd_index, i] = ((compare_ldds[head[0], head[1]] / noise_sample_ablating_results[head[0], head[1]]).item())

    return imshow(
        results,
        #facet_col = 0,
        #labels = [f"Head {head}" for head in key_backup_heads],
        title=f"The {'Ratio of Backup (Logit Diff Diff)' if not just_logits else 'Logit Diff Diffs'} of Intervention" + ("to Sample Ablation Backup" if not just_logits else ""),
        labels={"x": "Receiver Head", "y": "Intervention", "color": "Ratio of Logit Diff Diff to Sample Ablation" if not just_logits else "Logit Diff Diff"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        # range of y-axis color from 0 to 2
        #color_continuous_scale="mint",
        color_continuous_midpoint=1 if not just_logits else 0,
        # give x-axis labels
        x = [str(head) for head in heads],
        y = ldds_names,
        border=True,
        width=900,
        height = 600,
        margin={"r": 100, "l": 100},
        # show the values of the results above the heatmap
        text_auto = True,
        return_fig = True
    )

# %% [markdown]
# get results from replacing all IO directions

# %%

def run_interventions(return_just_lds = False):
    target_heads = [(9,6), (9,9)]#, (10,0)]

    
    zero_ablate_all_heads_ldds = project_stuff_on_heads(target_heads, project_only = True, scale_proj = 0, output = "get_ldd", freeze_ln=ln_on, return_just_lds = return_just_lds)
    project_only_io_direction = project_stuff_on_heads(target_heads, project_only = True, scale_proj = 1, output = "get_ldd", freeze_ln=ln_on, return_just_lds = return_just_lds)
    project_away_io_direction = project_stuff_on_heads(target_heads, project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=ln_on, return_just_lds = return_just_lds)

    
    model.reset_hooks()

    if ln_on:
        for layer in [9,10,11]:
            model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
            model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)
        model.add_hook("ln_final.hook_scale", patch_last_ln)

    for head in target_heads:

        # get the output of head on CORRUPTED RUN
        W_O_temp = model.W_O[head[0], head[1]]
        layer_z = corrupted_cache[utils.get_act_name("z", head[0])]
        layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
        output_head = layer_result[:, -1, head[1], :]

        # get projection of CORRUPTED HEAD OUTPUT onto IO token
        corrupted_head_only_IO_output = get_projection(output_head, target_intervene_direction)

        # add hook to now replace with this corrupted IO direction
        model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = target_intervene_direction, heads = [head[1]], replace_vector = corrupted_head_only_IO_output))

    replace_with_new_IO_logits, replace_with_new_IO_cache = model.run_with_cache(clean_tokens)

    model.reset_hooks()

    model.reset_hooks()
    if ln_on:
        for layer in [9,10,11]:
                    model.add_hook("blocks." + str(layer) + ".ln1.hook_scale", patch_ln_scale)
                    model.add_hook("blocks." + str(layer) + ".ln2.hook_scale", patch_ln2_scale)

    model.add_hook("ln_final.hook_scale", patch_last_ln)
    for head in target_heads:

        # get the output of head on CORRUPTED RUN
        W_O_temp = model.W_O[head[0], head[1]]
        layer_z = corrupted_cache[utils.get_act_name("z", head[0])]
        layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
        output_head = layer_result[:, -1, head[1], :]


        # get projection of CORRUPTED HEAD OUTPUT onto IO perp token
        corrupted_head_only_IO_output = get_projection(output_head, target_intervene_direction)
        everything_else_but_that = output_head - corrupted_head_only_IO_output

        # add hook to now replace with this corrupted IO perp direction
        model.add_hook(utils.get_act_name("result", head[0]), partial(project_away_component_and_replace_with_something_else, project_away_vector = target_intervene_direction, heads = [head[1]], replace_vector = everything_else_but_that, project_only = True))

    replace_with_new_perp_IO_logits, replace_with_new_perp_IO_cache = model.run_with_cache(clean_tokens)



    model.reset_hooks()

    if return_just_lds:
      replace_all_IOs_ldds = calc_all_logit_diffs(replace_with_new_IO_cache)[0] - ca
      replace_all_perp_IOs_ldds = calc_all_logit_diffs(replace_with_new_perp_IO_cache)[0] - ca
      return [zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds, project_away_io_direction, replace_all_IOs_ldds]
    else:

      replace_all_IOs_ldds = calc_all_logit_diffs(replace_with_new_IO_cache)[0]
      replace_all_perp_IOs_ldds = calc_all_logit_diffs(replace_with_new_perp_IO_cache)[0]
      return [zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds, project_away_io_direction, replace_all_IOs_ldds]


# %%
third_intervention = run_interventions(return_just_lds = True)
zero_ablate_all_heads_lds, project_only_io_direction_lds, replace_all_perp_IOs_lds, project_away_io_direction, replace_all_IOs_lds = third_intervention

#per_head_logit_diff, ablated_logit_diff

# %%
import plotly

# %%


# %%
fig = px.scatter()
x =  per_head_logit_diff.flatten()

# left
y =  replace_all_perp_IOs_lds.flatten()
fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(), text = fig_names, textposition="top center", mode = 'markers+text', name = "Replace IO-Perp Directions"))
x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
# right

y =  replace_all_IOs_lds.flatten()
fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(), text = fig_names, textposition="top center", mode = 'markers+text', name = "Replace IO Directions"))


# on both
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ))
  #y =  ablated_logit_diff.flatten()
  #fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(),  textposition="top center", mode = 'markers+text', name = "sample ablated", marker=dict(color="purple")), row = 1, col = col)



fig.update_xaxes(title = "Clean Direct Effect")
fig.update_yaxes(title = "Ablated Direct Effect")
fig.update_layout(title = "Logit Differences When Zero Ablating in Name Mover Heads", width = 950)
fig.show()

# %%

fig = px.scatter()
x =  per_head_logit_diff.flatten()

# left
y =  project_only_io_direction_lds.flatten()
fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(), text = fig_names, textposition="top center", mode = 'markers+text', name = "Only include IO"))
x_range = np.linspace(start=min(fig.data[1].x) - 0.5, stop=max(fig.data[1].x) + 0.5, num=100)
# right

y =  project_away_io_direction.flatten()
fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(), text = fig_names, textposition="top center", mode = 'markers+text', name = "Only include IO-perp"))


# on both
fig.add_trace(go.Scatter(x=x_range, y=x_range, mode='lines', name='y=x', line_color = "black", ))
  #y =  ablated_logit_diff.flatten()
  #fig.add_trace(go.Scatter(x = x.cpu(), y = y.cpu(),  textposition="top center", mode = 'markers+text', name = "sample ablated", marker=dict(color="purple")), row = 1, col = col)



fig.update_xaxes(title = "Clean Direct Effect")
fig.update_yaxes(title = "Ablated Direct Effect")
fig.update_layout(title = "Logit Differences When Zero Ablating in Name Mover Heads", width = 950)
fig.show()

# %%
# average across all three interventions
all_interventions = [third_intervention]#, second_intervention, new_intervention]
average_interventions = []
for i in range(len(third_intervention)):
    average_interventions.append((third_intervention[i]))

# %% [markdown]
# get results from replacing all perp to IO directions

# %%
fig = compare_intervention_ldds_with_sample_ablated([ca - ca] + average_interventions  + [noise_sample_ablating_results],
                                               ["Clean Run", "Zero Ablation of NMHs", "Project Only IO Direction (Zero ⊥ IO direction)", "Replace ⊥ IO directions with Corrupted ⊥ IO directions", "Project Away IO Direction (Zero IO direction)", "Replace IO directions with Corrupted IO directions",  "Sample Ablation of NMHs"],
                                               heads = key_backup_heads + neg_m_heads, just_logits = True)

# %%
# draw a rectangle in the fig
fig.add_shape(
    # unfilled Rectangle
        type="rect",
        x0=-0.5,
        y0=-0.5,
        x1=3.5,
        y1=6.49,
        line=dict(
            color="Black",
            #linewidth = 3
        ),
        #fillcolor="RoyalBlue",
        opacity=1,
        layer="above",

    )



fig.add_shape(
    # unfilled Rectangle
        type="rect",
        x0=3.5,
        y0=-0.5,
        x1=5.5,
        y1=6.49,
        line=dict(
            color="Black",
            #linewidth = 3
        ),
        #fillcolor="RoyalBlue",
        opacity=1,
        layer="above",

    )

# add text above the rectangle but on plot (not in cells)
# fig.add_annotation(

#         x=0,
#         y=0,
#         text="Interventions",
#         showarrow=False,
#         yshift=10,
# )

fig.show()

# %%
# for index, i in enumerate([zero_ablate_all_heads_ldds, project_only_io_direction, replace_all_perp_IOs_ldds,  project_away_io_direction, replace_all_IOs_ldds, noise_sample_ablating_results]):
#     imshow(i,
#            title = ["Zero Ablation of NMHs", "Project Only IO Direction (Zero ⊥ IO direction)", "Replace ⊥ IO directions with Corrupted ⊥ IO directions", "Project Away IO Direction (Zero IO direction)", "Replace IO directions with Corrupted IO directions",  "Sample Ablation of NMHs"][index])

# %%
# find cosine similarity of 9.0 output and IO unembedding
for head in neg_m_heads:
    print("Mean Cossim similarity between IO unembedding and " + str(head) + ":")
    W_O_temp = model.W_O[head[0], head[1]]
    layer_z = clean_cache[utils.get_act_name("z", head[0])]
    layer_result = einops.einsum(W_O_temp, layer_z, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
    output_head = layer_result[:, -1, head[1], :]

    # get projection of CORRUPTED HEAD OUTPUT onto IO token
    corrupted_head_only_IO_output = compute_cosine_similarity(output_head, target_intervene_direction)
    print(corrupted_head_only_IO_output.mean(0))

# %%


# %% [markdown]
# ### How much does Copy Suppression explain Self-Repair in the Negative Heads?

# %%
cache_patching_NMHs = act_patch(
    model = model,
    orig_input = clean_tokens,
    new_cache = corrupted_cache,
    patching_nodes = [Node("z", layer = 9 , head = 6) , Node("z", layer = 9 , head = 9), Node("z", layer = 10 , head = 0)],
    patching_metric = return_item,
    verbose = False,
    apply_metric_to_cache = True
)

patched_NMHs_logit_diff ,_,_ = calc_all_logit_diffs(cache_patching_NMHs)


# sum over last two layers of backup, but not including 10.0
patched_NMHS_backup = patched_NMHs_logit_diff - per_head_logit_diff
assert patched_NMHS_backup[:9].flatten().sum() == 0
actual_backup_CRS = patched_NMHS_backup.flatten().sum() - patched_NMHS_backup[10, 0] - patched_NMHS_backup[9,9] - patched_NMHS_backup[9,6]
self_repair_in_negative_heads = patched_NMHS_backup[10, 7] + patched_NMHS_backup[11,10]
print("Perent self repair explained by negative heads: ", self_repair_in_negative_heads / actual_backup_CRS)

# %%
model.set_use_attn_result(True)
# sum over last two layers of backup
all_NMH_project_no_io = project_stuff_on_heads(name_mover_heads, project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=True, return_just_lds = True)
project_stuff_on_heads(name_mover_heads, project_only = False, scale_proj = 1, output = "display_logits", freeze_ln=True, return_just_lds = True)
model.set_use_attn_result(False)


just_projection_self_repair = all_NMH_project_no_io - per_head_logit_diff
assert just_projection_self_repair[:9].flatten().sum().abs() <= 0.001
just_projection_CRS = just_projection_self_repair.flatten().sum() - just_projection_self_repair[10, 0] - just_projection_self_repair[9,6] - just_projection_self_repair[9,9]
just_projection_negative_head_self_repair = just_projection_self_repair[10, 7] + just_projection_self_repair[11,10]



# %%
print(f"Negative Heads make this much of Self-Repair: {self_repair_in_negative_heads / actual_backup_CRS}")

# %%
print(f"Copy Suppression explains how much in Negative Heads: {just_projection_negative_head_self_repair / self_repair_in_negative_heads}")

# %%
just_projection_negative_head_self_repair / actual_backup_CRS

# %% [markdown]
# # Static experiments

# %%
!git fetch https://github.com/callummcdougall/SERI-MATS-2023-Streamlit-pages/blob/d2a6ca671b9023abcdfb024c216418119b327e2e/transformer_lens/rs/callum2/explore_prompts/model_results_3.py#L275C1-L275C1

# %%
def get_effective_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid_pre = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(resid_pre)
    attn_out = einops.einsum(
        pre_attention, 
        model.W_V[0],
        model.W_O[0],
        "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
    )
    resid_mid = attn_out + resid_pre
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    W_ONLY_MLP = resid_pre.squeeze() + model.blocks[0].mlp(model.blocks[0].ln2(resid_pre)).squeeze()

    torch.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_U": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
        "Cody MLP": W_ONLY_MLP,
    }

# %%
embeddings = get_effective_embedding(model)

# %%
def combined_1_acc_iteration(full_OV_circuit: FactoredMatrix, top = True):
  actual_matrix = full_OV_circuit.AB
  top_sum = 0
  min_sum = 0

  #print(actual_matrix.shape[0])
  for col in range(actual_matrix.shape[0]):
    
    column = actual_matrix[:, col]
    top_sum += 1 if (column.argmax() == col).item() else 0
    min_sum += 1 if (column.argmin() == col).item() else 0
  return (top_sum - min_sum) / actual_matrix.shape[0]



names_list =  saved_names

# %%
def lock_attn(
    attn_patterns: Float[torch.Tensor, "batch head_idx dest_pos src_pos"],
    hook: HookPoint,
) -> Float[torch.Tensor, "batch head_idx dest_pos src_pos"]:
    print("LOCKING")
    assert isinstance(attn_patterns, Float[torch.Tensor, "batch head_idx dest_pos src_pos"]) # ensure shape is correct
    assert hook.layer() == 0 # only do this on layer 0
    batch, n_heads, seq_len = attn_patterns.shape[:3]

    attn_new = einops.repeat(torch.eye(seq_len), "dest src -> batch head_idx dest src", batch=batch, head_idx=n_heads).clone().to(attn_patterns.device)
    return attn_new

# %%
embeddings["Cody MLP"].shape

# %%
model.to_tokens(names, prepend_bos=False).shape

# %%
def look_at_backup_circuit(
    model: HookedTransformer,
    head_one: Tuple[int, int],
    head_two: Tuple[int, int],
    names,
    show_matrix = True,
    negative = False,
    both = False # use a combined metric which does both positive and negative backup at once
):
    """
    Shows the strength of the backup - W_OV^A, W_QK^B circuit - between heads
    or, the negative backup if negative = True
    """

    # Define components from our model (for typechecking, and cleaner code)
    embed = model.embed
    mlp0 = model.blocks[0].mlp
    ln0 = model.blocks[0].ln2
    unembed = model.unembed
    ln_final = model.ln_final

    # # Get embeddings for the names in our list
    # name_tokens: Int[Tensor, "batch 1"] = model.to_tokens(names, prepend_bos=False)
    # name_embeddings: Int[Tensor, "batch 1 d_model"] = embed(name_tokens)

    # # Get residual stream after applying MLP
    # resid_after_mlp1 = name_embeddings + mlp0(ln0(name_embeddings)) # seq 1 d_model
    # resid_after_mlp1 = resid_after_mlp1[:, 0, :]

    # Get MLP Embeddings
    name_tokens: Int[Tensor, "batch 1"] = model.to_tokens(names, prepend_bos=False)
    #print(name_tokens.shape)
    #embed_plus_MLP = embeddings["W_E (including MLPs)"]
    embed_plus_MLP = embeddings["Cody MLP"]
    name_embeddings = embed_plus_MLP[name_tokens][:, 0, :]
    #print(name_embeddings.shape)


    # calculate the OV matrix of head two
    A_O = model.W_O[head_one[0], head_one[1]]
    A_V = model.W_V[head_one[0], head_one[1]]
    A_OV_Circuit = FactoredMatrix(A_V, A_O)

    # calculate the QK matrix of head two
    B_Q = model.W_Q[head_two[0], head_two[1]]
    B_K = model.W_K[head_two[0], head_two[1]]


    if negative and not both:
      B_QK_Circuit = FactoredMatrix(-B_Q, B_K.T)
    else:
      B_QK_Circuit = FactoredMatrix(B_Q, B_K.T)


    relationship = A_OV_Circuit @ B_QK_Circuit # this is the A by B compositioin we want

    # put token embeddings around this matrix
    full_circuit = name_embeddings @ relationship @ name_embeddings.T

    # we got to find a way to combine both these metrics into one
    # fortunately, one easy way of doing this is just adding one if it is top_1, subtracting one if it is bottom_1
    top_1 = combined_1_acc_iteration(full_circuit)


    if show_matrix:
      print("THiS HAS NOT BEEN TESTED")
      print(top_1)
      print(top_5)
      imshow (
          full_circuit.AB,
          labels={"x": "Input token", "y": "Attention to output token"},
          title="Full Backup composition between head " + str(head_one) +" and "+ str(head_two),
          width=700,
          x = model.to_str_tokens(name_tokens),
          y = model.to_str_tokens(name_tokens)
      )
    else:
      return top_1

# %%
def display_back_scores(B_layer, B_head, negative = False, both = False):
  """
  displays all backup scores of heads with the head (B_layer, B_head)
  """

  backup_circuitry_11_7 = torch.zeros((12, 12))
  for layer in range(12):
    for head in range(12):
      
      top_1 = look_at_backup_circuit(model, (layer,head), (B_layer, B_head), names, show_matrix = False, negative = negative, both = both)

      
      backup_circuitry_11_7[layer][head] = top_1

  imshow (
          backup_circuitry_11_7,
          labels={"x": "Head", "y": "Layer"},
          title=f"Backup Circuit Score with {B_layer}.{B_head}" if not negative else f"Negative Backup Circuit Score with {B_layer}.{B_head}",
          width=700,
          range_color = [-1, 1]
      )

# %%
display_back_scores(10, 0, True)

# %%
def gather_backup_scores_between_heads(ov_head_list, qk_head_list, negative = False, both = False):
  """
  gathers backup scores between heads in a list
  """

  scores = torch.zeros((len(ov_head_list), len(qk_head_list)))
  for i, head_i in enumerate(ov_head_list):
    for j, head_j in enumerate(qk_head_list):
      top1 =  look_at_backup_circuit(model, head_i, head_j, names_list, show_matrix=False, negative = negative,both = both)

      if head_i[0] < head_j[0]:
        scores[i][j] = top1
      else:
        scores[i][j] = np.nan


  return scores

# %%
all_heads_list =  [(10,7), (11,10)] + [(9,9), (9,6), (10,0)] + [(10,2), (10,6), (10,10), (11,2)] + [(10,7), (11,10)]
#all_heads_list = [all_heads_list[-1 - i] for i in range(len(all_heads_list))]
ov_heads_list = [head for head in all_heads_list if head[0] == 9] + [(9,1)]
qk_heads_list = [head for head in all_heads_list if head[0] != 9] + [(11,0)]




interesting_head_backup_scores = gather_backup_scores_between_heads(ov_heads_list, qk_heads_list, negative = False, both = True)

# %%
interesting_head_backup_scores.shape

# %%
fig = imshow(
        interesting_head_backup_scores,
        return_fig = True,
        title="Static Backup Identity Scores between Key Heads",
        x = [str(i) for i in qk_heads_list], y = [str(i) for i in ov_heads_list],
        labels={"x": "QK Head", "y": "OV Head", "color": "Similarity to ± Identity"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=800,
        #margin={"r": 100, "l": 100},
        color_continuous_scale = "RdBu",
        #midpoint = 0,
    )

# fig.add_shape(type="rect",
#     x0=-0.5, y0=-0.5, x1=6.5, y1=1.5,
#     fillcolor="white",
#               layer='below',
#     line=dict(color="white"),
# )

fig.update_layout(
    #font_family="Courier New",
    #font_color="blue",
    #title_font_family="Times New Roman",
    #title_font_color="red",
    legend_title_font_color="green",
    #height =400
)
fig.show()

# %%


# %% [markdown]
# # What percent of downstream heads have strong backup with above?

# %%
all_heads_list =  [(10,7), (11,10)] + [(9,9), (9,6), (10,0)] + [(10,2), (10,6), (10,10), (11,2)] + [(10,7), (11,10)]
#all_heads_list = [all_heads_list[-1 - i] for i in range(len(all_heads_list))]
ov_heads_list = name_mover_heads#[[9,i] for i in range(12)] + [[10,i] for i in range(12)] + [[11,i] for i in range(12)]
qk_heads_list = [[10,i] for i in range(12)] + [[11,i] for i in range(12)] 




interesting_head_backup_scores = gather_backup_scores_between_heads(ov_heads_list, qk_heads_list, negative = False, both = True)

# %%
fig = imshow(
        interesting_head_backup_scores,
        return_fig = True,
        title="Static Backup Identity Scores between Key Heads",
        x = [str(i) for i in qk_heads_list], y = [str(i) for i in ov_heads_list],
        labels={"x": "QK Head", "y": "OV Head", "color": "Similarity to ± Identity"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        border=True,
        width=800,
        #margin={"r": 100, "l": 100},
        color_continuous_scale = "RdBu",
        #midpoint = 0,
    )

fig.add_shape(type="rect",
    x0=-0.5, y0=1.5, x1=11.5, y1=2.5,
    fillcolor="black",
              layer='above',
    # no line
    line = dict(
        color="black",
        width=0,
    ),
    # have shape be on top
    #xref='x', yref='y',
    
)

fig.update_layout(
    #font_family="Courier New",
    #font_color="blue",
    #title_font_family="Times New Roman",
    #title_font_color="red",
    legend_title_font_color="green",
    width = 1000,
    height = 350
)
fig.show()

# %%
fig.write_image("sbis_scores.pdf")

# %%


# %%



