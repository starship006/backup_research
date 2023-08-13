# %%

# !sudo apt install unzip
# !pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
# !pip install git+https://github.com/neelnanda-io/neel-plotly.git
# !pip install 
#!pip install plotly fancy_einsum jaxtyping transformers datasets transformer_lens
from imports import *
from different_nmh_dataset_gen import generate_dataset
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

TOTAL_TYPES = 4
NUM_PROMPTS = 10 * 6 * TOTAL_TYPES
PROMPTS_PER_TYPE = int(NUM_PROMPTS / TOTAL_TYPES)
PROMPTS, CORRUPT_PROMPTS, ANSWERS, INCORRECT_ANSWERS, TYPE_NAMES = generate_dataset(model, NUM_PROMPTS, TOTAL_TYPES)


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

def calc_all_logit_contibutions(cache, per_prompt = False, include_incorrect = INCLUDE_INCORRECT, include_MLP = False) -> Float[Tensor, "layer head"]:
  clean_per_head_residual, labels = cache.stack_head_results(layer = -1, return_labels = True, apply_ln = False) # per_head_residual.shape = heads batch seq_pos d_model
  #print(clean_per_head_residual.shape)
  # also, for the worried, no, we're not missing the application of LN here since it gets applied in the below function call
  per_head_logit_diff: Float[Tensor, "batch head"] = residual_stack_to_logit_diff(clean_per_head_residual[:, :, -1, :], clean_cache, per_prompt=per_prompt, include_incorrect=include_incorrect)

  clean_per_layer_residual = torch.zeros((model.cfg.n_layers, clean_per_head_residual.shape[1], clean_per_head_residual.shape[2], clean_per_head_residual.shape[3])).to(device)
  #print(clean_per_layer_residual.shape)
  
  for layer in range(model.cfg.n_layers):
      clean_per_layer_residual[layer] = cache[f'blocks.{layer}.hook_mlp_out']

  #print(clean_per_layer_residual.shape)
  per_layer_logit_diff: Float[Tensor, "layer"] = residual_stack_to_logit_diff(clean_per_layer_residual[:, :, -1, :], clean_cache, per_prompt=per_prompt, include_incorrect=include_incorrect)
  #print(per_layer_logit_diff.shape)
  #print(per_layer_logit_diff)

  per_head_logit_diff = einops.rearrange(
      per_head_logit_diff,
      "(layer head) ... -> layer head ...",
      layer=model.cfg.n_layers
  )

  # concatenate MLP results at the end of head results
  if include_MLP:
    if per_prompt:
        enargened_logit_diffs = torch.zeros((model.cfg.n_layers, model.cfg.n_heads + 1, per_layer_logit_diff.shape[-1])).to(device)
        for layer in range(model.cfg.n_layers):
            enargened_logit_diffs[layer][0:model.cfg.n_heads] = per_head_logit_diff[layer]
            enargened_logit_diffs[layer][model.cfg.n_heads] = per_layer_logit_diff[layer]
    else:
        print("NOT TESTED")
        return
        per_head_logit_diff = torch.cat((per_head_logit_diff, per_layer_logit_diff), 1)
  
  #print(per_head_logit_diff.shape)
  #print(per_head_logit_diff[:, -1, 3])
  if include_MLP:
    return enargened_logit_diffs
  else:
      return per_head_logit_diff





# def display_all_logits(cache = None, per_head_ld = None, title = "Logit Contributions", comparison = False,
#                         return_fig = False, logits = None, include_incorrect = INCLUDE_INCORRECT):
#     """
#     given an input, display the logit contributions of each head

#     comparison: if True, display logit contribution/diff diff; if False, display logits contibution/diff
#     """
    
#     if per_head_ld is not None and cache is None:
#         assert per_head_ld.shape == (model.cfg.n_layers, model.cfg.n_heads, NUM_PROMPTS)
#     if per_head_ld is not None and cache is not None:
#         # throw error - only one should be passed
#         raise ValueError("Only one of per_head_ld and cache should be passed")

#     if logits is not None:
#         ld = logits_to_ave_logit_diff(logits, include_incorrect=include_incorrect)
#     else:
#         ld = 0.00

#     if per_head_ld is not None:
#         a = per_head_ld
#     else:
#         a = calc_all_logit_contibutions(cache, per_prompt=True, include_incorrect=include_incorrect)
#     if not comparison:
#         a_1, a_2, a_3, a_4 = a[..., :PROMPTS_PER_TYPE], a[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], a[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], a[..., 3*PROMPTS_PER_TYPE:]
#         fig = imshow(
#             torch.stack([(a_1).mean(-1), 
#                             (a_2).mean(-1), (a_3).mean(-1), (a_4).mean(-1)]),
#             return_fig = True,
#             facet_col = 0,
#             facet_labels = TYPE_NAMES,
#             title=title,
#             labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
#             #coloraxis=dict(colorbar_ticksuffix = "%"),
#             border=True,
#             width=1500,
#             margin={"r": 100, "l": 100},
#         )
#     else:
#         ca = calc_all_logit_contibutions(clean_cache, per_prompt = True, include_incorrect=include_incorrect)
#         a_1, a_2, a_3, a_4 = a[..., :PROMPTS_PER_TYPE], a[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], a[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], a[..., 3*PROMPTS_PER_TYPE:]
#         ca_1, ca_2, ca_3, ca_4 = ca[..., :PROMPTS_PER_TYPE], ca[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], ca[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], ca[..., 3*PROMPTS_PER_TYPE:]

#         #print(a_1.shape, ca_1.shape, a_2.shape, ca_2.shape, a_3.shape, ca_3.shape, a_4.shape, ca_4.shape)
#         assert a_1.shape == ca_1.shape == a_2.shape == ca_2.shape == a_3.shape == ca_3.shape == a_4.shape == ca_4.shape

#         fig = imshow(
#             torch.stack([(a_1 - ca_1).mean(-1), 
#                             (a_2-ca_2).mean(-1), (a_3 - ca_3).mean(-1), (a_4 - ca_4).mean(-1)]),
#             return_fig = True,
#             facet_col = 0,
#             facet_labels = TYPE_NAMES,
#             title=title,
#             labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
#             #coloraxis=dict(colorbar_ticksuffix = "%"),
#             border=True,
#             width=1700,
#             margin={"r": 100, "l": 100}
#         )

#     if return_fig:
#         return fig
#     else:
#         fig.show()



def display_all_logits(cache = None, per_head_ld = None, title = "Logit Contributions", comparison = False,
                        return_fig = False, logits = None, include_incorrect = INCLUDE_INCORRECT):
    """
    given an input, display the logit contributions of each head. this version also displays the MLP layer!!

    comparison: if True, display logit contribution/diff diff; if False, display logits contibution/diff
    """
    if per_head_ld is not None and cache is None:
        assert per_head_ld.shape == (model.cfg.n_layers, model.cfg.n_heads + 1, NUM_PROMPTS)
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
        a = calc_all_logit_contibutions(cache, per_prompt=True, include_incorrect=include_incorrect, include_MLP=True)


    # with TYPE_NAMES above each subplot
    fig = plotly.subplots.make_subplots(rows = 1, cols = TOTAL_TYPES, subplot_titles = TYPE_NAMES, shared_yaxes=True)# subplot_titles = TYPE_NAMES)
    fig.update_layout(yaxis_autorange="reversed")
    index = 0
    
    a_1, a_2, a_3, a_4 = a[..., :PROMPTS_PER_TYPE], a[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], a[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], a[..., 3*PROMPTS_PER_TYPE:]
    if comparison:
        ca = calc_all_logit_contibutions(clean_cache, per_prompt = True, include_incorrect=include_incorrect, include_MLP=True)
        ca_1, ca_2, ca_3, ca_4 = ca[..., :PROMPTS_PER_TYPE], ca[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], ca[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], ca[..., 3*PROMPTS_PER_TYPE:]
        
        print(a_1.shape)
        print(ca_1.shape)
        a_1 = a_1 - ca_1
        a_2 = a_2 - ca_2
        a_3 = a_3 - ca_3
        a_4 = a_4 - ca_4



    #a_1, a_2, a_3, a_4 = a[..., :PROMPTS_PER_TYPE], a[..., PROMPTS_PER_TYPE:2*PROMPTS_PER_TYPE], a[..., 2*PROMPTS_PER_TYPE:3*PROMPTS_PER_TYPE], a[..., 3*PROMPTS_PER_TYPE:]
    for a in [a_1, a_2, a_3, a_4 ]:
        # add heatmap trace
        to_graph = (a).mean(-1).cpu()
        fig.add_trace(
            go.Heatmap(
                z = to_graph,
                x = list(range(model.cfg.n_heads + 13)),
                y = list(range(model.cfg.n_layers)),
                colorscale = "RdBu",
                colorbar = dict(
                    title = "Logit Contribution",
                    
                ),
                
                coloraxis = "coloraxis",
            ),
            row = 1,
            col = index + 1
        )
        x = 12
        y = -0.5
        width = 1
        depth = 12
        fig.add_shape(
            type = "rect",
            x0 = x,
            x1 = x + width,
            y0 = y,
            y1 = y + depth,
            line = dict(color = "black"),
            row = 1,
            col = index + 1
        )
        

        # add sideways annotation of text
        fig.add_annotation(
            x = 12.5,
            y = 5.5,
            text = "MLP",
            showarrow = False,
            textangle = 270,
            row = 1,
            col = index + 1
        )
        index += 1
    
    fig.update_coloraxes(colorscale="RdBu", colorbar_title="Logit Contribution", cmid = 0)
    fig.update_layout(
        title = title,
    )
    fig.update_xaxes(title_text = "Head")
    fig.update_yaxes(title_text = "Layer")
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

    results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads + 1, NUM_PROMPTS)).cuda()

    for receiver_layer in range(model.cfg.n_layers):
        for receiver_head in range(model.cfg.n_heads):
            per_head_logit_diff = path_patch(model, clean_tokens, corrupt_tokens,
                                    [Node("z", layer, head) for (layer,head) in ablate_heads],
                                    [Node("q", receiver_layer, receiver_head)],
                                     partial(calc_all_logit_contibutions, per_prompt=True, include_incorrect=include_incorrect, include_MLP = True),
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

unembed_io_directions = model.tokens_to_residual_directions(answer_tokens[:, 0]).to(device)
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


def project_stuff_on_heads(project_heads, direction = target_intervene_direction, project_only = False, scale_proj = 1, output = "display_logits", freeze_ln = False, return_just_lds = False):

    model.reset_hooks()
    # project_heads is a list of tuples (layer, head). for each layer, write a hook which projects all the heads from the layer
    for layer in range(model.cfg.n_layers):
        key_heads = [head[1] for head in project_heads if head[0] == layer]
        if len(key_heads) > 0:
            #print(key_heads)
            model.add_hook(utils.get_act_name("result", layer), partial(project_vector_operation, vector = direction, heads = key_heads, scale_proj = scale_proj, project_only = project_only))

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

clone_layer = 9
into_layer = 10
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
into_layer = 10
heads_to_write_into = [(into_layer,j) for j in range(12)] #+  [(11,i) for i in range(12)]

for act, act_name in [("q", "query"), ("k", "key"),  ("v", "value")]:
    model.reset_hooks()
    resid_pre = clean_cache[utils.get_act_name("resid_pre", clone_layer)].clone()
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
heads_whose_outputs_to_simulate =  [(9,i) for i in range(12) if (9,i) not in []]
resid_pre = clean_cache[utils.get_act_name("resid_pre", clone_layer)].clone()
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
# %% More experiments involving allowing certain heads to go through and contribute to the MLP layer, as well.
clone_layer = 9
l2_of_clone_layer = model.blocks[clone_layer].ln2
mlp_of_clone_layer = model.blocks[clone_layer].mlp

into_layer = 10
heads_to_write_into = [(into_layer,j) for j in range(12)]

heads_whose_outputs_to_add_and_mlpify = [(9,i) for i in range(12) if (9,i) not in [(9,6), (9,9)]]#[(9,6), (9,8), (9,9), (9,4), (9,5),(9,0)] #[(9,i) for i in range(12) if (9,i) not in []]
heads_whose_outputs_to_not_mlpify = [(9,i) for i in range(12) if (9,i) not in heads_whose_outputs_to_add_and_mlpify]

resid_pre_for_mlp = clean_cache[utils.get_act_name("resid_pre", clone_layer)].clone()
attn_into_mlp_out = torch.zeros(resid_pre_for_mlp.shape).cuda()
attn_fake_out = torch.zeros(resid_pre_for_mlp.shape).cuda()

for head in heads_whose_outputs_to_add_and_mlpify:
    attn_into_mlp_out += clean_cache[utils.get_act_name("result", head[0])][:, :, head[1], :]
if True:
    attn_into_mlp_out += model.b_O[clone_layer]


for heads in heads_whose_outputs_to_not_mlpify:
    attn_fake_out += clean_cache[utils.get_act_name("result", heads[0])][:, :, heads[1], :]

resid_mid_for_mlp = resid_pre_for_mlp + attn_into_mlp_out
l2_resid_pre_for_mlp = l2_of_clone_layer(resid_mid_for_mlp)
resid_post_mlp = resid_mid_for_mlp + mlp_of_clone_layer(l2_resid_pre_for_mlp)

# add in head outputs that we didnt mlpify
resid_post_mlp += attn_fake_out
for act, act_name in [("q", "query"), ("k", "key"),  ("v", "value")]:
    model.reset_hooks()
    for head in heads_to_write_into:
        model.add_hook(utils.get_act_name(act, head[0]), partial(kqv_rewrite_hook, changed_resid = resid_post_mlp, head = head, act_name = act))
    _, hooked_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    # display new logit diff
    display_all_logits(hooked_cache, comparison = True, title = f"Logit Diff Diff from cloning resid_pre_{clone_layer} into the {act_name} of downstream layers, while simulating output of {heads_whose_outputs_to_add_and_mlpify} and mlpifying\n and not mlpifying everything else", include_incorrect=INCLUDE_INCORRECT)

# %% Subtracting 9.6 and 9.9 direct, indirect, and both

clone_layer = 9
l2_of_clone_layer = model.blocks[clone_layer].ln2
mlp_of_clone_layer = model.blocks[clone_layer].mlp

into_layer = 10
heads_to_write_into = [(into_layer,j) for j in range(12)]

ablation_results =[] # direct, indirect, both
ablate_heads = [(9,6), (9,9)]

for ablate_direct, ablate_indirect in [(True, False), (False, True), (True, True)]:

    if ablate_direct and ablate_indirect:
        heads_whose_outputs_to_add_and_mlpify = [(9,i) for i in range(12) if (9,i) not in ablate_heads]
        heads_whose_outputs_to_not_mlpify = []
    elif ablate_direct:
        # have everything initially go through MLP, then subtract out the direct effect later
        heads_whose_outputs_to_add_and_mlpify = [(9,i) for i in range(12) if (9,i) not in []]
        heads_whose_outputs_to_not_mlpify = [(9,i) for i in range(12) if (9,i) not in heads_whose_outputs_to_add_and_mlpify]    
    elif ablate_indirect:
        heads_whose_outputs_to_add_and_mlpify = [(9,i) for i in range(12) if (9,i) not in ablate_heads]
        heads_whose_outputs_to_not_mlpify = [(9,i) for i in range(12) if (9,i) not in heads_whose_outputs_to_add_and_mlpify]

    resid_pre_for_mlp = clean_cache[utils.get_act_name("resid_pre", clone_layer)].clone()
    attn_into_mlp_out = torch.zeros(resid_pre_for_mlp.shape).cuda()
    attn_fake_out = torch.zeros(resid_pre_for_mlp.shape).cuda()

    for head in heads_whose_outputs_to_add_and_mlpify:
        attn_into_mlp_out += clean_cache[utils.get_act_name("result", head[0])][:, :, head[1], :]
    if True:
        attn_into_mlp_out += model.b_O[clone_layer]


    for heads in heads_whose_outputs_to_not_mlpify:
        attn_fake_out += clean_cache[utils.get_act_name("result", heads[0])][:, :, heads[1], :]

    resid_mid_for_mlp = resid_pre_for_mlp + attn_into_mlp_out
    l2_resid_pre_for_mlp = l2_of_clone_layer(resid_mid_for_mlp)
    resid_post_mlp = resid_mid_for_mlp + mlp_of_clone_layer(l2_resid_pre_for_mlp)

    # INTERVENTION: delete directions only if ablating direct (only)
    if ablate_direct and not ablate_indirect:
        resid_post_mlp -= clean_cache[utils.get_act_name("result", 9)][:, :, 6, :] + clean_cache[utils.get_act_name("result", 9)][:, :, 9, :]

    # add in head outputs that we didnt mlpify
    resid_post_mlp += attn_fake_out
    for act, act_name in [("q", "query"), ("k", "key"),  ("v", "value")]:
        model.reset_hooks()
        for head in heads_to_write_into:
            model.add_hook(utils.get_act_name(act, head[0]), partial(kqv_rewrite_hook, changed_resid = resid_post_mlp, head = head, act_name = act))
        _, hooked_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()
        # display new logit diff
        display_all_logits(hooked_cache, comparison = True, title = f"Logit Diff Diff from cloning resid_pre_{clone_layer} into the {act_name} of downstream layers, while simulating output of {heads_whose_outputs_to_add_and_mlpify} and mlpifying\n and not mlpifying everything else", include_incorrect=INCLUDE_INCORRECT)

        if act_name == "query":
            ablation_results.append(calc_all_logit_contibutions(hooked_cache, per_prompt=True, include_incorrect=INCLUDE_INCORRECT, include_MLP=True) - calc_all_logit_contibutions(clean_cache, per_prompt=True, include_incorrect=INCLUDE_INCORRECT, include_MLP=True))

# %% to what extent can you describe the total change in logit diff from ablating everything as ldd from ablating direct + ldd from ablating indirect effect

for head in range(12):
    print(f"10.{head}: {(ablation_results[0][10, head].mean(0) + ablation_results[1][10, head]).mean(0) / ablation_results[2][10, head].mean(0)}")
# %% learn a vector to try to see if we can recover the direction which activates backup in the non CS heads
torch.set_grad_enabled(True)
def get_output_of_head(layer, head, cache = clean_cache):
    matrix = model.W_O[layer, head]
    layer_output = cache[utils.get_act_name("z", layer)]
    layer_result = einops.einsum(matrix, layer_output, "d_head d_model, batch seq h_idx d_head -> batch seq h_idx d_model")
    output_of_head = layer_result[:, -1, head, :]
    return output_of_head

def apply_causal_mask(
        attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        '''
        Applies a causal mask to attention scores, and returns masked scores.
        '''
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = torch.ones(attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device)
        mask = torch.triu(all_ones, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, 1e-6)
        return attn_scores


def simulate_head(resid, layer, head):
    """
    given a specific head in the model, simulates running a normalized residual stream through it
    note that this does the normal gpt-2 style head. not the fancy stuff.
    """

    # get parameters for head
    W_Q = model.W_Q[layer, head]
    W_K = model.W_K[layer, head]
    W_V = model.W_V[layer, head]
    W_O = model.W_O[layer, head]

    b_Q = model.b_Q[layer, head]
    b_K = model.b_K[layer, head]
    b_V = model.b_V[layer, head]
    #b_O = model.b_O[layer, head]

    # calculate keys, queries, and values
    Q = einops.einsum(resid, W_Q, "batch seq d_model, d_model d_head -> batch seq d_head") + b_Q
    K = einops.einsum(resid, W_K, "batch seq d_model, d_model d_head -> batch seq d_head") + b_K
    V = einops.einsum(resid, W_V, "batch seq d_model, d_model d_head -> batch seq d_head") + b_V

    # calculate attention scores
    attn_scores = einops.einsum(
            Q, K,
            "batch posn_Q d_head, batch posn_K d_head -> batch posn_Q posn_K",
        )

    # mask
    attn_scores = apply_causal_mask(attn_scores/ model.cfg.d_head ** 0.5)

    # calculate attention probs
    attn_probs = F.softmax(attn_scores, dim=-1)

    # weighted sum of values, according to attention probs
    z = einops.einsum(
            V, attn_probs,
            "batch posn_K d_head, batch posn_Q posn_K -> batch posn_Q d_head",
        )

    attn_out = einops.einsum(
            z, W_O,
            "batch posn_Q d_head, d_head d_model -> batch posn_Q d_model")#+ b_O

    return attn_out



# %%
resid_mid_layer_nine = clean_cache[utils.get_act_name("resid_mid", 9)]

nine_nine_sample_ablating_cache = dir_effects_from_sample_ablating_head([(9,9)])

sample_ablate_ten_zero = get_output_of_head(10, 0, nine_nine_sample_ablating_cache)
sample_ablate_ten_two = get_output_of_head(10, 2, nine_nine_sample_ablating_cache)
sample_ablate_ten_six = get_output_of_head(10, 6, nine_nine_sample_ablating_cache)
sample_ablate_ten_ten = get_output_of_head(10, 10, nine_nine_sample_ablating_cache)

clean_ten_zero = get_output_of_head(10, 0)
clean_ten_two = get_output_of_head(10, 2)
clean_ten_six = get_output_of_head(10, 6)
clean_ten_ten = get_output_of_head(10, 10)

nine_nine_output = get_output_of_head(9, 9, clean_cache)
nine_six_output = get_output_of_head(9, 6, clean_cache)
# %%
def backup_loss_metric(current_head_output, ablated_output, abs = True):
    # find logit diff of current head output and ablated output
    assert current_head_output.shape == ablated_output.shape
    cur_head_logit_diff = residual_stack_to_logit_diff(current_head_output, nine_nine_sample_ablating_cache, include_incorrect= INCLUDE_INCORRECT)
    ablated_logit_diff = residual_stack_to_logit_diff(ablated_output, nine_nine_sample_ablating_cache, include_incorrect=INCLUDE_INCORRECT)

    if abs:
        return (ablated_logit_diff - cur_head_logit_diff).abs()
    else:
        return ablated_logit_diff - cur_head_logit_diff

def perp_to_IO_metric(vector, direction = unembed_io_directions):
    assert len(direction.shape) == 2
    if len(vector.shape) != 2:
        vector = einops.repeat(vector, "d_model -> batch d_model", batch = direction.shape[0]).clone().to(device)
    return F.cosine_similarity(vector, unembed_io_directions, dim=-1).mean(0).abs()

def normalize_metric(vector):
    # push down norm of vector
    return einops.einsum(vector, vector, "d_model, d_model -> ") / 100

# %% start by trying to train a single vector that we hope controls for backup across all batch examples
if True:
  learned_vectors = []



for i in tqdm(range(4)):
    trained_vector = torch.randn(model.cfg.d_model, requires_grad=True, device = device)
    optimizer = torch.optim.Adam([trained_vector], lr=0.1)

    for i in range(700):
        optimizer.zero_grad()
        repeated_learned_vector = einops.repeat(trained_vector, "d_model -> batch d_model", batch = nine_nine_output.shape[0])

        # part 1 -- the removing vector should ACTIVATE backup
        new_residual_stream = resid_mid_layer_nine.clone()
        new_residual_stream[:, -1, :] = new_residual_stream[:, -1, :] - get_projection(nine_nine_output, repeated_learned_vector)

        new_residual_stream = new_residual_stream + model.blocks[9].mlp(model.blocks[9].ln2(new_residual_stream))
        new_residual_stream = model.blocks[10].ln1(new_residual_stream)



        # part 2 -- simulate new head outputs
        ten_two_output = simulate_head(new_residual_stream, 10, 2)[:, -1, :]
        ten_six_output = simulate_head(new_residual_stream, 10, 6)[:, -1, :]
        ten_ten_output = simulate_head(new_residual_stream, 10, 10)[:, -1, :]
        ten_zero_output = simulate_head(new_residual_stream, 10, 0)[:, -1, :]

        # part 3 -- if you constrain the heads output to ONLY this, it shouldn't activate backup
        second_residual_stream = resid_mid_layer_nine.clone()
        second_residual_stream[:, -1, :] = second_residual_stream[:, -1, :] - nine_nine_output + get_projection(nine_nine_output, repeated_learned_vector)
        second_residual_stream = second_residual_stream + model.blocks[9].mlp(model.blocks[9].ln2(second_residual_stream))
        second_residual_stream = model.blocks[10].ln1(second_residual_stream)

        ten_two_output_second = simulate_head(second_residual_stream, 10, 2)[:, -1, :]
        ten_six_output_second = simulate_head(second_residual_stream, 10, 6)[:, -1, :]
        ten_ten_output_second = simulate_head(second_residual_stream, 10, 10)[:, -1, :]
        ten_zero_output_second = simulate_head(second_residual_stream, 10, 0)[:, -1, :]


        # part 4 - get losses on other constraints
        perp_io_loss = perp_to_IO_metric(repeated_learned_vector, direction = unembed_io_directions)
        activate_loss = backup_loss_metric(ten_two_output, ablated_output = sample_ablate_ten_two) + backup_loss_metric(ten_six_output, ablated_output = sample_ablate_ten_six) + backup_loss_metric(ten_ten_output, ablated_output = sample_ablate_ten_ten) + backup_loss_metric(ten_zero_output, ablated_output = sample_ablate_ten_zero)
        silence_loss = backup_loss_metric(ten_two_output_second, ablated_output = clean_ten_two) + backup_loss_metric(ten_six_output_second, ablated_output = clean_ten_six) + backup_loss_metric(ten_ten_output_second, ablated_output = clean_ten_ten) + backup_loss_metric(ten_zero_output_second, ablated_output = clean_ten_zero)

        loss = activate_loss + 2 * silence_loss + perp_io_loss + normalize_metric(trained_vector)

        # part 5 - standard training stuff
        loss.backward()
        optimizer.step()
    
    learned_vectors.append(trained_vector.detach().cpu())

torch.set_grad_enabled(False)
# %% calculate the cosine similarities between the vectors in learned_vectors

cosine_similarities = torch.zeros((len(learned_vectors), len(learned_vectors)))
for i in range(len(learned_vectors)):
     for j in range(len(learned_vectors)):
        cosine_similarities[i][j] = (F.cosine_similarity(learned_vectors[i], learned_vectors[j], dim=0))

imshow(cosine_similarities, title = "cos sims between the learned 'backup vectors'", width = 600
       )
# %% get length of trained vector
for vec in learned_vectors:
  print(einops.einsum(vec, vec, "d_model, d_model -> "))
print(einops.einsum(nine_nine_output, nine_nine_output, "batch d_model, batch d_model -> batch").mean(0))
# %% get perpendicularity to IO direction
for vec in learned_vectors:
  print(perp_to_IO_metric(vec))
# %% Display a bunch of these ldds given the intervention, in relation to other ones
key_backup_heads = [(10,0), (10,2), (10,6), (10,10)]
def compare_intervention_ldds_with_sample_ablated(all_ldds, ldds_names, heads = key_backup_heads, just_logits = False):
    results = torch.zeros((len(all_ldds), len(heads)))


    if just_logits:
        for ldd_index, compare_ldds in enumerate(all_ldds):
            for i, head in enumerate(heads):
                #print(head)
                results[ldd_index, i] = ((compare_ldds[head[0], head[1]]).item()) # / noise_sample_ablating_results[head[0], head[1]]).item())
    else:
        pass
        # for ldd_index, compare_ldds in enumerate(all_ldds):
        #     for i, head in enumerate(heads):
        #         #print(head)
        #         results[ldd_index, i] = ((compare_ldds[head[0], head[1]] / noise_sample_ablating_results[head[0], head[1]]).item())

    return imshow(
        results,
        #facet_col = 0,
        #labels = [f"Head {head}" for head in key_backup_heads],
        title=f"The {'Ratio of Backup (Logit Diff Diff)' if not just_logits else 'Logit Diff Diffs'} of Intervention" + ("to Sample Ablation Backup" if not just_logits else ""),
        labels={"x": "Receiver Head", "y": "Intervention", "color": "Ratio of Logit Diff Diff to Sample Ablation" if not just_logits else "Logit Diff Diff"},
        #coloraxis=dict(colorbar_ticksuffix = "%"),
        # range of y-axis color from 0 to 2
        #color_continuous_scale="mint",
        #color_continuous_midpoint=1 if not just_logits else 0,
        # give x-axis labels
        x = [str(head) for head in heads],
        y = ldds_names,
        border=True,
        width=900,
        height = 50 * len(all_ldds) + 300,
        margin={"r": 100, "l": 100},
        # show the values of the results above the heatmap
        text_auto = True,
        return_fig = True
    )

first_intervention = run_interventions()
# %%
just_mystery_vectors = []
away_mystery_vectors = []

for vec in learned_vectors:
  repeated_trained_vector = einops.repeat(vec, "d_model -> batch d_model", batch = nine_nine_output.shape[0]).to(device)

  just_mystery_vectors.append(project_stuff_on_heads([(9,9)], repeated_trained_vector, project_only = True, scale_proj = 1, output = "get_ldd", freeze_ln=True))
  away_mystery_vectors.append(project_stuff_on_heads([(9,9)], repeated_trained_vector, project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=True))

just_9_6 = project_stuff_on_heads( [(9,9)], nine_six_output, project_only = True, scale_proj = 1, output = "get_ldd", freeze_ln=True)
just_9_9 = project_stuff_on_heads([(9,9)], nine_nine_output, project_only = True, scale_proj = 1, output = "get_ldd", freeze_ln=True)
away_just_9_6 = project_stuff_on_heads([(9,9)], nine_six_output, project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=True)
away_just_9_9 = project_stuff_on_heads([(9,9)], nine_nine_output, project_only = False, scale_proj = 1, output = "get_ldd", freeze_ln=True)
# %%

activate_interventions = {
    "Clean Run" : calc_all_logit_contibutions(clean_cache, per_prompt = False, include_incorrect=INCLUDE_INCORRECT) - calc_all_logit_contibutions(clean_cache, per_prompt = False, include_incorrect=INCLUDE_INCORRECT),
    #"Sample Ablation of NMHs" : noise_sample_ablating_results,
    "Sample Ablation of (9.6)" : calc_all_logit_contibutions(nine_nine_sample_ablating_cache, per_prompt = False, include_incorrect=INCLUDE_INCORRECT) - calc_all_logit_contibutions(clean_cache, per_prompt = False, include_incorrect=INCLUDE_INCORRECT),
    "Zero Ablation of NMHs" : first_intervention[0],
    #"Project Only IO Direction (Zero ⊥ IO direction)" : first_intervention[1],
    #"Replace ⊥ IO directions with Corrupted ⊥ IO directions" : first_intervention[2],
    #"Project Away IO Direction (Zero IO direction)" : first_intervention[3],
    #"Replace IO directions with Corrupted IO directions" : first_intervention[4],


    "Project Only (9.6)" : just_9_6,
    "Project Away (9.6)" : away_just_9_6,
    "Project Only (9.9)" : just_9_9,
    "Project Away (9.9)" : away_just_9_9,
    #"Projecting Only (9.6 ∥ 9.9) ∥ IO" : ldd_from_96_99_feature_and_parra_to_io,
    #"Projecting Away (9.6 ∥ 9.9) ∥ IO" : ldd_from_away_96_99_feature_and_parra_to_io,
    #"Project Only (9.6 ∥ 9.9)" : overlap_9_6_9_9,
    #"Project Away (9.6 ∥ 9.9)" : away_overlap_9_6_9_9,
    #"Just the Learned Vector" : just_mystery_vector,
    #"Project Away the Learned Vector" : away_mystery_vector
}

for i, results in enumerate(just_mystery_vectors):
    activate_interventions["just " + str(i)] = results

for i, results in enumerate(away_mystery_vectors):
    activate_interventions["away " + str(i)] = results


# %%
row_labels = [str(i) for i in activate_interventions.keys()]
values = [activate_interventions[key] for key in activate_interventions.keys()]

fig = compare_intervention_ldds_with_sample_ablated(values, row_labels, heads = key_backup_heads , just_logits = True) # + neg_m_heads
fig.show()
# %%
