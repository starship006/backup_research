"""
This file extends upon the Backup tooling to view how backup accurs across the distribution on MLPs.
"""
# %%
# !pip install git+https://github.com/neelnanda-io/TransformerLens.git
# !sudo apt install unzip
# !pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
# !pip install git+https://github.com/neelnanda-io/neel-plotly.git
# !pip install plotly
from imports import *

# %%
model_name = "gpt2-small"
#backup_storage_file_name = model_name + "_new_backup_count_storage.pickle"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
    device = device,
)
# %% Dataest
owt_dataset = utils.get_dataset("owt")
BATCH_SIZE = 30
PROMPT_LEN = 40

all_owt_tokens = model.to_tokens(owt_dataset[0:BATCH_SIZE * 2]["text"]).to(device)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)

# %%

torch.cuda.empty_cache()

# %%

logits, cache = model.run_with_cache(owt_tokens)

print(utils.lm_accuracy(logits, owt_tokens))
print(utils.lm_cross_entropy_loss(logits, owt_tokens))
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
        
def show_input(input_head_values: Float[Tensor, "n_layer n_head"], input_MLP_layer_values: Float[Tensor, "n_layer"]):
    """
    greats a figure with the head values displayed on the left, with the MLP values displayed on the right
    """



    fig = make_subplots(rows = 1, cols = 2, subplot_titles = ["Heads", "MLP"])

    fig.add_trace(go.Heatmap(
        z = input_head_values.cpu(),
        x = list(range(input_head_values.shape[1])),
        y = list(range(input_head_values.shape[0])),
        colorbar = dict(title = "Logit Contribution"),
        colorscale = "RdBu",
        coloraxis="coloraxis",
    ), row = 1, col = 1)

    # reshape MLP tensor
    input_MLP_layer_values = einops.repeat(input_MLP_layer_values, "n_layer -> n_layer 1")
    fig.add_trace(go.Heatmap(
        z = input_MLP_layer_values.cpu(),
        x = list(range(input_MLP_layer_values.shape[1])),
        y = list(range(input_MLP_layer_values.shape[0])),
        colorbar = dict(title = "Logit Contribution"),
        colorscale = "RdBu",

        coloraxis="coloraxis"
    ), row = 1, col = 2)

    # reverse the y axis
    fig.update_yaxes(autorange="reversed")

    # update coloraxis to be RdBU
        
    max_value = max(np.abs(input_head_values.cpu()).max(), np.abs(input_MLP_layer_values.cpu()).max()).item()
    fig.update_layout(coloraxis=dict(colorscale="RdBu", cmin=-max_value, cmax=max_value))


    fig.show()

def collect_direct_effect(cache: ActivationCache, correct_tokens: Float[Tensor, "batch seq_len"],
                           title = "Direct Effect of Heads", display = True) -> Tuple[Float[Tensor, "n_layer n_head batch pos_minus_one"], Float[Tensor, "n_layer d_mlp batch pos_minus_one"]]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head and neuron on each token.
    
    returns tuple of tensors ([heads, batch, pos - 1], [n_layer d_mlp batch pos_minus_one] length tensor of direct effects of each head on each (correct) token

    cache: cache of activations from the model
    correct_tokens: [batch, seq_len] tensor of correct tokens
    title: title of the plot (relavant if display == True)
    display: whether to display the plot or return the data; if False, returns [head, pos] tensor of direct effects
    """

    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(correct_tokens)
    
    # get the direct effect of heads by positions
    clean_per_head_residual: Float[Tensor, "head batch seq d_model"] = cache.stack_head_results(layer = -1, return_labels = False, apply_ln = False)
    print(clean_per_head_residual.shape)
    per_head_direct_effect: Float[Tensor, "heads batch pos_minus_one"] = residual_stack_to_direct_effect(clean_per_head_residual,
                                                                                          cache, token_residual_directions,
                                                                                          batch_pos_dmodel = True, average_across_batch = False,
                                                                                          apply_ln = True)
    per_head_direct_effect = einops.rearrange(per_head_direct_effect, "(n_layer n_head) batch pos -> n_layer n_head batch pos", n_layer = model.cfg.n_layers, n_head = model.cfg.n_heads)
    #assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1])

    # get the outputs of the neurons
    direct_effect_mlp: Float[Tensor, "n_layer d_mlp batch pos_minus_one"] = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp, owt_tokens.shape[0], owt_tokens.shape[1] - 1))
    
    # iterate over every neuron to avoid memory issues
    for neuron in tqdm.tqdm(range(model.cfg.d_mlp)):
        single_neuron_output: Float[Tensor, "n_layer batch pos d_model"] = cache.stack_neuron_results(layer = -1, neuron_slice = (neuron, neuron + 1), return_labels = False, apply_ln = False)
        direct_effect_mlp[:, neuron, :, :] = residual_stack_to_direct_effect(single_neuron_output,
                                                                                          cache, token_residual_directions,
                                                                                          batch_pos_dmodel = True, average_across_batch = False,
                                                                                          apply_ln = True).cpu()
    

    # get per mlp layer effect
    all_layer_output: Float[Tensor, "n_layer batch pos d_model"] = torch.zeros((model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1], model.cfg.d_model)).cuda()
    for layer in range(model.cfg.n_layers):
        all_layer_output[layer, ...] = cache[f'blocks.{layer}.hook_mlp_out']

    all_layer_direct_effect: Float["n_layer batch pos_minus_one"] = residual_stack_to_direct_effect(all_layer_output,
                            cache, token_residual_directions, batch_pos_dmodel = True, average_across_batch = False,
                            apply_ln = True).cpu()

    # temp_all_layer_direct_effect = direct_effect_mlp.sum(dim = 1).cpu()
    # see if the two are close
    # print(all_layer_direct_effect.shape, temp_all_layer_direct_effect.shape)
    # print("All layer direct effect close?", torch.allclose(all_layer_direct_effect, temp_all_layer_direct_effect, atol = 0.2))
    # print(all_layer_direct_effect[0,0])
    # print(temp_all_layer_direct_effect[0,0])



    if display:    
        mean_per_head_direct_effect = per_head_direct_effect.mean(dim = (-1, -2))
        
        imshow(
            torch.stack([mean_per_head_direct_effect]),
            return_fig = False,
            facet_col = 0,
            facet_labels = [f"Direct Effect of Heads"],
            title=title,
            labels={"x": "Head", "y": "Layer", "color": "Logit Contribution"},
            #coloraxis=dict(colorbar_ticksuffix = "%"),
            border=True,
            width=500,
            margin={"r": 100, "l": 100}
        )
        

        
    
    return per_head_direct_effect, direct_effect_mlp, all_layer_direct_effect
    
def return_item(item):
  return item

# %%
torch.cuda.empty_cache()
per_head_direct_effect, per_neuron_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, owt_tokens, display = True)
torch.cuda.empty_cache()
# %%
print(per_neuron_direct_effect.shape)

# %%
show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)))

# %%
