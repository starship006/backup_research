# %%
from imports import *

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


# %% Dataest
owt_dataset = utils.get_dataset("owt")
BATCH_SIZE = 100
PROMPT_LEN = 150

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


# %%
def view_top_prompts_by_head(layer, head, topk = 10, view_negative_contributions = False):
    head_direct_effects = per_head_direct_effect[layer, head]
    top_indicies = topk_of_Nd_tensor(head_direct_effects, topk) if not view_negative_contributions else topk_of_Nd_tensor(-head_direct_effects, topk)
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

        
        print("----------------------------------------------------------------------------")
        print(f"Layer {layer}, Head {head}, Batch {batch}, Pos {pos}")
        print("\n Logit Contribution: " + str(per_head_direct_effect[layer, head, batch, pos].item()) + "\n Backup Amount: " + str(backup_amounts[index]))
        print(prompt_up_till_word + "  ---------> " + predicted_word)
        index += 1
        
# %%
view_top_prompts_by_head(10,7, topk = 40, view_negative_contributions = True)
# %%
batch = 12

plotly.io.renderers.default = "vscode"
cv.activations.text_neuron_activations(
    model.to_str_tokens(owt_tokens[batch, 1:]),
    einops.rearrange(per_head_direct_effect[:, :, batch, :], "layer head pos -> pos layer head"),
    second_dimension_name = "Head"
)

# %%

batch = 84

plotly.io.renderers.default = "vscode"
cv.activations.text_neuron_activations(
    model.to_str_tokens(owt_tokens[batch, 1:]),
    einops.rearrange(per_head_direct_effect[:, :, batch, :], "layer head pos -> pos layer head"),
    second_dimension_name = "Head"
)
# %%

# %%
def stare_at_attention_and_head_pat(cache, layer_to_stare_at, head_to_isolate, display_corrupted_text = False, verbose = True, specific = False, specific_index = 0):
  """
  given a cache from a run, displays the attention patterns of a layer, as well as printing out how much the model
  attends to the S1, S2, and IO token
  """

  tokenized_str_tokens =  model.to_str_tokens(owt_tokens[specific_index])
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


#   print("Attention on S1: " + str(S1))
#   print("Attention on IO: " + str(IO))
#   print("Attention on S2: " + str(S2))
#   print("S1 + IO - S2 = " + str(S1 + IO - S2))
#   print("S1 + S2 - IO = " + str(S1 + S2 - IO))
#   print("S1 - IO - S2 = " + str(S1 - S2 - IO))


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

# %%
stare_at_attention_and_head_pat(cache, 10, 7, specific = True, specific_index = 84, verbose = False)
# %%
