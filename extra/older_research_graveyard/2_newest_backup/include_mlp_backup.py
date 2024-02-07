"""
This file extends upon the Backup tooling to view how backup accurs across the distribution on MLPs.
"""
# %%
# !sudo apt install unzip
# !pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
# !pip install git+https://github.com/neelnanda-io/neel-plotly.git
# !pip install plotly
#from TransformerLens import transformer_lens
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
BATCH_SIZE = 50
PROMPT_LEN = 50

all_owt_tokens = model.to_tokens(owt_dataset[0:BATCH_SIZE * 2]["text"]).to(device)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)

# %%

torch.cuda.empty_cache()
model.set_use_attn_result(True)

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
                           title = "Direct Effect of Heads", display = True, collect_individual_neurons = False) -> Tuple[Float[Tensor, "n_layer n_head batch pos_minus_one"], Float[Tensor, "n_layer batch pos_minus_one"], Float[Tensor, "n_layer d_mlp batch pos_minus_one"]]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head and neuron on each token.
    
    returns tuple of tensors of per-head, per-mlp-layer, per-neuron of direct effects

    cache: cache of activations from the model
    correct_tokens: [batch, seq_len] tensor of correct tokens
    title: title of the plot (relavant if display == True)
    display: whether to display the plot or return the data; if False, returns [head, pos] tensor of direct effects
    """

    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(correct_tokens)
    
    # get the direct effect of heads by positions
    clean_per_head_residual: Float[Tensor, "head batch seq d_model"] = cache.stack_head_results(layer = -1, return_labels = False, apply_ln = False)
    #print(clean_per_head_residual.shape)
    per_head_direct_effect: Float[Tensor, "heads batch pos_minus_one"] = residual_stack_to_direct_effect(clean_per_head_residual,
                                                                                          cache, token_residual_directions,
                                                                                          batch_pos_dmodel = True, average_across_batch = False,
                                                                                          apply_ln = True)
    per_head_direct_effect = einops.rearrange(per_head_direct_effect, "(n_layer n_head) batch pos -> n_layer n_head batch pos", n_layer = model.cfg.n_layers, n_head = model.cfg.n_heads)
    #assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1])

    # get the outputs of the neurons
    direct_effect_mlp: Float[Tensor, "n_layer d_mlp batch pos_minus_one"] = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp, owt_tokens.shape[0], owt_tokens.shape[1] - 1))
    
    # iterate over every neuron to avoid memory issues
    if collect_individual_neurons:
        for neuron in tqdm(range(model.cfg.d_mlp)):
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
        

    per_head_direct_effect = per_head_direct_effect.to(device)
    all_layer_direct_effect = all_layer_direct_effect.to(device)
    direct_effect_mlp = direct_effect_mlp.to(device)
        
    if collect_individual_neurons:
        return per_head_direct_effect, all_layer_direct_effect, direct_effect_mlp
    else:
        return per_head_direct_effect, all_layer_direct_effect
    
def return_item(item):
  return item

# %%
torch.cuda.empty_cache()
per_head_direct_effect, all_layer_direct_effect, per_neuron_direct_effect  = collect_direct_effect(cache, owt_tokens, display = True, collect_individual_neurons = True)
torch.cuda.empty_cache()
# %%
print(per_neuron_direct_effect.shape)

# %%
show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)))

# %%

def dir_effects_from_sample_ablating(attention_heads = None, mlp_layers = None, neurons = None):
    """this function gets the new direct effect of all the heads when sample ablating the input head
    it uses the global cache, owt_tokens, corrupted_owt_tokens

    attention_heads: list of tuples of (layer, head) to ablate
    mlp_layers: list of layers to ablate
    neurons: list of tuples of (layer, neuron) to ablate
    """

    # don't accept if more than one input is none
    assert sum([attention_heads is not None, mlp_layers is not None, neurons is not None]) == 1
    
    if attention_heads is not None:
        new_cache = act_patch(model, owt_tokens, [Node("z", layer, head) for (layer,head) in attention_heads],
                            return_item, corrupted_owt_tokens, apply_metric_to_cache= True)
    elif mlp_layers is not None:
        new_cache = act_patch(model, owt_tokens, [Node("mlp_out", layer) for layer in mlp_layers],
                            return_item, corrupted_owt_tokens, apply_metric_to_cache= True)
    elif neurons is not None:
        new_cache = act_patch(model, owt_tokens, [Node("post", layer = layer, neuron = neuron) for (layer,neuron) in neurons],
                            return_item, corrupted_owt_tokens, apply_metric_to_cache= True)

        
    head_direct_effect, mlp_layer_direct_effect = collect_direct_effect(new_cache, owt_tokens, display = False, collect_individual_neurons = False)
                                            
    return head_direct_effect, mlp_layer_direct_effect

# %%
temp_head_effect, temp_mlp_effect =dir_effects_from_sample_ablating(attention_heads = [(10,7)])
show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), 
           temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)))
# %%
temp_head_effect, temp_mlp_effect =dir_effects_from_sample_ablating(mlp_layers= [10])
show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), 
           temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)))
# %%
temp_head_effect, temp_mlp_effect =dir_effects_from_sample_ablating(neurons = [(2,1)])
show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), 
           temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)))
# %% see if any ablation of heads causes positive change in direct effect of components
pairs_of_head_backup = {}
pairs_of_mlp_backup = {}
threshold = 0.05
for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        temp_head_effect, temp_mlp_effect =dir_effects_from_sample_ablating(attention_heads = [(layer, head)])
        head_backup: Float[Tensor, "layer head"] = temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2))
        mlp_backup: Float[Tensor, "layer"] = temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2))

        if (head_backup > threshold).sum() > 0:
            pairs_of_head_backup[(layer, head)] = True
        else:
            pairs_of_head_backup[(layer, head)] = False

        if (mlp_backup > threshold).sum() > 0:
            pairs_of_mlp_backup[layer] = True
        else:
            pairs_of_mlp_backup[layer] = False
# %%
for layer, head in pairs_of_head_backup.keys():
    if pairs_of_head_backup[(layer, head)]:
        print((layer, head))

    if pairs_of_mlp_backup[layer]:
        print(layer)
# %%

# # see if any ablation of neurons causes positive change in direct effect of components
# new_pairs_of_head_backup = {}
# new_pairs_of_mlp_backup = {}
# threshold = 0.05
# for layer in tqdm(range(model.cfg.n_layers)):
#     for neuron in range(model.cfg.d_mlp):
#         temp_head_effect, temp_mlp_effect =dir_effects_from_sample_ablating(neurons = [(layer, neuron)])
#         head_backup: Float[Tensor, "layer head"] = temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2))
#         mlp_backup: Float[Tensor, "layer"] = temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2))

#         if (head_backup > threshold).sum() > 0:
#             new_pairs_of_head_backup[(layer, neuron)] = True
#         else:
#             new_pairs_of_head_backup[(layer, neuron)] = False

#         if (mlp_backup > threshold).sum() > 0:
#             new_pairs_of_mlp_backup[layer] = True
#         else:
#             new_pairs_of_mlp_backup[layer] = False
# # %%
# for layer, neuron in new_pairs_of_head_backup.keys():
#     if new_pairs_of_head_backup[(layer, neuron)]:
#         print((layer, neuron))

#     if new_pairs_of_mlp_backup[layer]:
#         print(layer)
        
# # %%


# %%

def create_scatter_of_backup_of_component(heads = None, mlp_layers = None, return_slope = False):
    """"
    this function:
    1) gets the direct effect of all a component when sample ablating it
    2) gets the total accumulated backup of the component for each prompt and position
    3) plots the clean direct effect vs accumulated backup

    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    """

    # don't accept if more than one input is none
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
    else:
        raise Exception("No heads or mlp layers given")
    
    ablated_per_head_batch_direct_effect, mlp_per_layer_direct_effect = dir_effects_from_sample_ablating(attention_heads=heads, mlp_layers=mlp_layers)

    # 2) gets the total accumulated backup of the head for each prompt and position
    downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
    downstream_change_in_mlp_logit_diff: Float[Tensor, "layer batch pos"] = mlp_per_layer_direct_effect - all_layer_direct_effect
    #print(downstream_change_in_logit_diff.shape)
    #print(downstream_change_in_mlp_logit_diff.shape)
    
    assert downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() == 0
    if heads is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer):].sum(0)
        total_backup = head_backup + mlp_backup
    if mlp_layers is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer+1):].sum(0)
        total_backup = head_backup + mlp_backup
    
    
    #  3) plots the clean direct effect vs accumulated backup

    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else all_layer_direct_effect[layer].flatten().cpu()
    assert direct_effects.shape == total_backup.flatten().cpu().shape
    
    if not return_slope:
        fig = go.Figure()
        scatter_plot = go.Scatter(
            x = direct_effects,
            y = total_backup.flatten().cpu(),
            text=[f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN))],  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Compensatory Response Size"
        )
        fig.add_trace(scatter_plot)

        second_scatter = go.Scatter(
            x = direct_effects,
            y = head_backup.flatten().cpu() ,
            text=[f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN))],  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Response Size of just Attention Blocks"
        )
        fig.add_trace(second_scatter)

        third_scatter = go.Scatter(
            x = direct_effects,
            y = mlp_backup.flatten().cpu() ,
            text=[f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN))],  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Response Size of just MLP Blocks"
        )
        fig.add_trace(third_scatter)


    # get a best fit line
   
    slope, intercept = np.linalg.lstsq(np.vstack([direct_effects, np.ones(len(direct_effects))]).T, total_backup.flatten().cpu(), rcond=None)[0]

    if not return_slope:
        max_x = max(direct_effects)

        # add line of best fit
        fig.add_trace(go.Scatter(
            x=torch.linspace(0,max_x,100),
            y=torch.linspace(0,max_x,100) * slope + intercept,
            mode='lines',
            name='lines'
        ))

        component = heads if heads is not None else mlp_layers
        fig.update_layout(
            title=f"Total Accumulated Backup of {component} in {model_name} for each Position and Batch" if heads is not None 
            else f"Total Accumulated Backup of MLP Layer {component} in {model_name} for each Position and Batch",
        )
        fig.update_xaxes(title = f"Direct Effect of Head {heads[0]}" if heads is not None else f"Direct Effect of MLP Layer {mlp_layers[0]}")
        fig.update_yaxes(title = "Total Accumulated Backup")
        fig.update_layout(width=900, height=500)
        fig.show()
    
    if return_slope:
        return slope

# %%
create_scatter_of_backup_of_component(mlp_layers = [4])
# %%
create_scatter_of_backup_of_component(heads = [(9,6)])
# %%


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

            ablated_per_head_batch_direct_effect, ablated_per_mlp_layer_direct_effect = dir_effects_from_sample_ablating([(layer, head)])
            downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
            downstream_change_in_mlp_logit_diff: Float[Tensor, "layer batch pos"] = ablated_per_mlp_layer_direct_effect - all_layer_direct_effect
            if topk_prompts > 0:
                backup_amount = 0.0
                for batch, pos in top_indices:
                    select_heads = downstream_change_in_logit_diff[(layer+1):, :, batch, pos]
                    select_mlp_layers = downstream_change_in_mlp_logit_diff[layer:, batch, pos]
                    # for the heads in select_heads, get the total downstream change
                    backup_amount += select_heads.sum((0,1)).item() + select_mlp_layers.sum().item()
                backup_amount /= topk_prompts
            else:
                # use all batch and prompts
                backup_amount = downstream_change_in_logit_diff[(layer+1):].mean((0,1,2,3)).item() + downstream_change_in_mlp_logit_diff[(layer):].mean((0,1,2)).item()
            
            
            total_accumulated_backup_per_head[layer, head] = backup_amount
    
    return total_accumulated_backup_per_head, average_clean_logit_diff
# %%


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
top_k_to_isolate = 40
total_accumulated_backup_per_head, direct_clean_effect_per_head = get_backup_per_head(top_k_to_isolate)
# %% Plot the results:
plot_accumulated_backup_per_head(top_k_to_isolate, total_accumulated_backup_per_head, direct_clean_effect_per_head)


def get_backup_per_layer(topk_prompts = 0):
    """
    gets the downstream accumulated backup when ablating an mlp layer
    by default, this operates across all prompts: if topk_prompts > 0, it isolates the top_k prompts where 
    the head has the highest direct effect

    also returns the clean logit diffs (either across all prompts, or on all top_k ones)
    """
    total_accumulated_backup_per_head = torch.zeros((model.cfg.n_layers, ))
    average_clean_logit_diff = torch.zeros((model.cfg.n_layers,))
    for layer in range(model.cfg.n_layers):
        if topk_prompts > 0:
            layer_direct_effects = all_layer_direct_effect[layer]
            top_indices = topk_of_Nd_tensor(layer_direct_effects, topk_prompts)
            # set average_clean_logit_diff
            sum = 0
            for batch, pos in top_indices:
                sum += all_layer_direct_effect[layer, batch, pos].item()
            average_clean_logit_diff[layer] = sum / topk_prompts
        else:
            average_clean_logit_diff[layer] = per_head_direct_effect[layer].mean((0,1)).item()

        ablated_per_head_batch_direct_effect, ablated_per_mlp_layer_direct_effect = dir_effects_from_sample_ablating(mlp_layers = [layer])
        downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
        downstream_change_in_mlp_logit_diff: Float[Tensor, "layer batch pos"] = ablated_per_mlp_layer_direct_effect - all_layer_direct_effect
        if topk_prompts > 0:
            backup_amount = 0.0
            for batch, pos in top_indices:
                select_heads = downstream_change_in_logit_diff[(layer+1):, :, batch, pos]
                select_mlp_layers = downstream_change_in_mlp_logit_diff[(layer+1):, batch, pos]
                # for the heads in select_heads, get the total downstream change
                backup_amount += select_heads.sum((0,1)).item() + select_mlp_layers.sum().item()
            backup_amount /= topk_prompts
        else:
            # use all batch and prompts
            backup_amount = downstream_change_in_logit_diff[(layer+1):].mean((0,1,2,3)).item() + downstream_change_in_mlp_logit_diff[(layer+1):].mean((0,1,2)).item()
        
        
        total_accumulated_backup_per_head[layer] = backup_amount

    return total_accumulated_backup_per_head, average_clean_logit_diff

def plot_accumulated_backup_per_layer(top_k_to_isolate, total_accumulated_backup_per_head, direct_clean_effect_per_head):
    fig = go.Figure()
    colors = ['rgb(0, 0, 0)'] * model.cfg.n_layers   # Initialize with black color
    group_size = 1
    num_groups = model.cfg.n_layers 
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

        title=f"Total Accumulated Backup of Layers on top {(str(round(top_k_to_isolate / (BATCH_SIZE * (PROMPT_LEN - 1)) * 100, 2)) +'%') if top_k_to_isolate != 0 else 'All'}"
        + " Prompts vs. Average Direct Effect of Layers",
        width = 1000
    )
    fig.update_xaxes(title = "Average Direct Effect of Layer")
    fig.update_yaxes(title = "Total Accumulated Backup")
    fig.show()
# %%


top_k_to_isolate = 40
total_accumulated_backup_per_layer, direct_clean_effect_per_layer = get_backup_per_layer(top_k_to_isolate)

# %%
plot_accumulated_backup_per_layer(top_k_to_isolate, total_accumulated_backup_per_layer, direct_clean_effect_per_layer)

# %%
slopes_of_head_backup = torch.zeros((12,12))
for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        slopes_of_head_backup[layer, head] = create_scatter_of_backup_of_component(heads = [(layer, head)], return_slope = True)
        
# %%

imshow(slopes_of_head_backup, title = "Slopes of Head Backup",
       text_auto = True, width = 800, height = 800)# show a number above each square)

# %%


def get_backup_per_neuron(topk_prompts = 0):
    """
    gets the downstream accumulated backup when ablating a neuron
    by default, this operates across all prompts: if topk_prompts > 0, it isolates the top_k prompts where 
    the neuron has the highest direct effect

    also returns the clean logit diffs (either across all prompts, or on all top_k ones)
    """
    total_accumulated_backup_per_neuron = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp))
    average_clean_logit_diff = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp))
    
    total_iterations = model.cfg.n_layers * model.cfg.d_mlp
    print("Starting to iterate")
    pbar = tqdm(total=total_iterations)
    for layer in range(model.cfg.n_layers):
        for neuron in range(model.cfg.d_mlp):
            if topk_prompts > 0:
                neuron_direct_effects = per_neuron_direct_effect[layer, neuron]
                top_indices = topk_of_Nd_tensor(neuron_direct_effects, topk_prompts)
                # set average_clean_logit_diff
                #print(top_indices)
                average_clean_logit_diff[layer, neuron] = sum(per_neuron_direct_effect[layer, neuron, batch, pos].item() for batch, pos in top_indices) / topk_prompts
            else:
                average_clean_logit_diff[layer, neuron] = per_neuron_direct_effect[layer].mean((0,1)).item()

            ablated_per_head_batch_direct_effect, ablated_per_mlp_layer_direct_effect = dir_effects_from_sample_ablating(neurons = [(layer, neuron)])
            downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
            downstream_change_in_mlp_logit_diff: Float[Tensor, "layer batch pos"] = ablated_per_mlp_layer_direct_effect - all_layer_direct_effect
            
            if topk_prompts > 0:
                backup_amount = 0.0

                backup_amount = sum(downstream_change_in_logit_diff[(layer+1):, :, batch, pos].sum((0,1)).item() + downstream_change_in_mlp_logit_diff[(layer+1):, batch, pos].sum().item() for batch, pos in top_indices) / topk_prompts

            else:
                # use all batch and prompts
                backup_amount = downstream_change_in_logit_diff[(layer+1):].mean((0,1,2,3)).item() + downstream_change_in_mlp_logit_diff[(layer+1):].mean((0,1,2)).item()
            
            
            total_accumulated_backup_per_neuron[layer, neuron] = backup_amount
            pbar.update(1)

    return total_accumulated_backup_per_neuron, average_clean_logit_diff
# %%

def plot_accumulated_backup_per_neuron(top_k_to_isolate, total_accumulated_backup_per_head, direct_clean_effect_per_head):
    fig = go.Figure()
    colors = ['rgb(0, 0, 0)'] * model.cfg.n_layers * model.cfg.d_mlp  # Initialize with black color
    group_size = model.cfg.d_mlp
    num_groups = model.cfg.n_layers * model.cfg.d_mlp // group_size
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
        text=[f"Layer {i[0]}, Neuron {i[1]}" for i in itertools.product(range(model.cfg.n_layers), range(model.cfg.d_mlp))],  # Set the hover labels to the text attribute
        mode='markers',
        marker=dict(size=10, color=colors, opacity=0.8),
    )
    fig.add_trace(scatter_plot)
    fig.update_layout(

        title=f"Total Accumulated Backup of Neuron on top {(str(round(top_k_to_isolate / (BATCH_SIZE * (PROMPT_LEN - 1)) * 100, 2)) +'%') if top_k_to_isolate != 0 else 'All'}"
        + " Prompts vs. Average Direct Effect of Neurons",
        width = 1000
    )
    fig.update_xaxes(title = "Average Direct Effect of Neuron")
    fig.update_yaxes(title = "Total Accumulated Backup")
    fig.show()



# %%
topk_prompts = 50
print(device)
top_accumulated_backup_per_neuron, average_clean_logit_diff_per_neuron = get_backup_per_neuron(topk_prompts = topk_prompts)
# %%

# save with pickle

# with open("abc.pkl", "wb") as f:
#     pickle.dump(top_accumulated_backup_per_neuron, f)



# with open(f"backup_per_neuron_{model_name}_{BATCH_SIZE}_{PROMPT_LEN}_{topk_prompts}.pkl", "wb") as f:
#     pickle.dump(top_accumulated_backup_per_neuron, f)

# %%
plot_accumulated_backup_per_neuron(10, top_accumulated_backup_per_neuron, average_clean_logit_diff_per_neuron)
# %%
