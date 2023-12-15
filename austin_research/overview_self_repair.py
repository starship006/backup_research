# %%
from imports import *
import argparse

# %%
in_notebook_mode = True
if in_notebook_mode:
    model_name = "pythia-160m"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name

safe_model_name = model_name.replace("/", "_")
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed = True, 
    center_writing_weights = True,
    fold_ln = True, # TODO; understand this
    refactor_factored_attn_matrices = False,
    device = device,
)
model.set_use_attn_result(True)
# %%
owt_dataset = utils.get_dataset("owt")
BATCH_SIZE = 100
PROMPT_LEN = 50


all_owt_tokens = model.to_tokens(owt_dataset[0:BATCH_SIZE * 2]["text"]).to(device)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)

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
    residual_stack: Float[Tensor, "... batch pos d_model"],
    effect_directions: Float[Tensor, "batch pos d_model"],
    apply_last_ln = True,
    clean_cache = cache
) -> Float[Tensor, "... batch pos_mins_one"]:
    '''
    Gets the direct effect towards a direction for a given stack of components in the residual stream.
    NOTE: IGNORES THE VERY LAST PREDICTION AND FIRST CLEAN TOKEN; WE DON'T KNOW THE ACTUAL PREDICTED ANSWER FOR IT!

    residual_stack: [... batch pos d_model] components of d_model vectors to measure direct effect from
    effect_directions: [batch pos d_model] vectors in d_model space that correspond to 'direct effect'
    clean_cache = a cache from the clean run to use for the LN
    '''
    batch_size = residual_stack.size(-3)
    scaled_residual_stack = clean_cache.apply_ln_to_stack(residual_stack, layer=-1, has_batch_dim=True) if apply_last_ln else residual_stack
    
    # remove the last prediction, and the direction of the zeroth token
    scaled_residual_stack = scaled_residual_stack[..., :, :-1, :]
    effect_directions = effect_directions[:, 1:, :]

    return einops.einsum(scaled_residual_stack, effect_directions, "... batch pos d_model, batch pos d_model -> ... batch pos")


# test to ensure functions work: compare the predicted direct effect from the residual stack function to the actual direct effect
last_layer_direct_effect: Float[Tensor, "batch pos_minus_one"] = residual_stack_to_direct_effect(
    cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"],
    model.tokens_to_residual_directions(owt_tokens),
    )

# first_batch_with_bias = [predicted_last_layer_direct_effect[0, i] + model.b_U[owt_tokens[0, i + 1].item()] for i in range(49)]
# # compare the predicted 'direct effect' to the logits on the correct token. these should be the same
# key_logits = [logits[0, index, token].item() for index, token in enumerate(owt_tokens[0, 1:])]

# # compare the predicted 'direct effect' to the logits on the correct token. these should be the same
# for i in range(10):
#     print(first_batch_with_bias[i], key_logits[i])

def show_input(input_head_values: Float[Tensor, "n_layer n_head"], input_MLP_layer_values: Float[Tensor, "n_layer"],
               title = "Values"):
    """
    creates a heatmap figure with the head values displayed on the left, with the last column on the right displaying MLP values
    """

    input_head_values_np = input_head_values.cpu().numpy()
    input_MLP_layer_values_np = input_MLP_layer_values.cpu().numpy()
    
    # Combine the head values and MLP values for the heatmap
    combined_data = np.hstack([input_head_values_np, input_MLP_layer_values_np[:, np.newaxis]])
    
    fig = go.Figure(data=go.Heatmap(z=combined_data, colorscale='RdBu', zmid=0))

    fig.add_shape(
        go.layout.Shape(
            type="rect",
            xref="x",
            yref="y",
            x0=combined_data.shape[1] - 1.5,
            x1=combined_data.shape[1] - 0.5,
            y0=-0.5,
            y1=combined_data.shape[0] - 0.5,
            line=dict(color="Black", width=2)
        )
    )

    fig.update_layout(title=title, xaxis_title="Head", yaxis_title="Layer", yaxis_autorange='reversed')
    fig.show()
    

    # max_value = max(np.abs(input_head_values.cpu()).max(), np.abs(input_MLP_layer_values.cpu()).max()).item()
    # fig.update_layout(coloraxis=dict(colorscale="RdBu", cmin=-max_value, cmax=max_value))
    

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
    per_head_direct_effect: Float[Tensor, "heads batch pos_minus_one"] = residual_stack_to_direct_effect(clean_per_head_residual, token_residual_directions, True)
    
    
    per_head_direct_effect = einops.rearrange(per_head_direct_effect, "(n_layer n_head) batch pos -> n_layer n_head batch pos", n_layer = model.cfg.n_layers, n_head = model.cfg.n_heads)
    #assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1])

    # get the outputs of the neurons
    direct_effect_mlp: Float[Tensor, "n_layer d_mlp batch pos_minus_one"] = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp, owt_tokens.shape[0], owt_tokens.shape[1] - 1))
    
    # iterate over every neuron to avoid memory issues
    if collect_individual_neurons:
        for neuron in tqdm(range(model.cfg.d_mlp)):
            single_neuron_output: Float[Tensor, "n_layer batch pos d_model"] = cache.stack_neuron_results(layer = -1, neuron_slice = (neuron, neuron + 1), return_labels = False, apply_ln = False)
            direct_effect_mlp[:, neuron, :, :] = residual_stack_to_direct_effect(single_neuron_output, token_residual_directions).cpu()
    # get per mlp layer effect
    all_layer_output: Float[Tensor, "n_layer batch pos d_model"] = torch.zeros((model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1], model.cfg.d_model)).cuda()
    for layer in range(model.cfg.n_layers):
        all_layer_output[layer, ...] = cache[f'blocks.{layer}.hook_mlp_out']

    all_layer_direct_effect: Float["n_layer batch pos_minus_one"] = residual_stack_to_direct_effect(all_layer_output, token_residual_directions).cpu()


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


def dir_effects_from_sample_ablating(attention_heads = None, mlp_layers = None, neurons = None, return_cache = False) -> Union[ActivationCache, Tuple[Float[Tensor, "heads batch pos_minus_one"], Float[Tensor, "n_layer batch pos_minus_one"]]]:
    """this function gets the new direct effect of all the heads when sample ablating the input head
    it uses the global cache, owt_tokens, corrupted_owt_tokens

    attention_heads: list of tuples of (layer, head) to ablate
    mlp_layers: list of layers to ablate
    neurons: list of tuples of (layer, neuron) to ablate
    return_cache: whether to return the cache as well as the direct effect
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
                                            
    if return_cache:
        return head_direct_effect, mlp_layer_direct_effect, new_cache
    else:
        return head_direct_effect, mlp_layer_direct_effect
# %%
# get per component direct effects
per_head_direct_effect, all_layer_direct_effect, per_neuron_direct_effect  = collect_direct_effect(cache, owt_tokens, display = in_notebook_mode, collect_individual_neurons = True)


# %%
# show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)))
show_input(per_head_direct_effect.mean((-1,-2)),torch.zeros((model.cfg.n_layers)))

# %%
histogram(per_head_direct_effect[7,8].flatten(), title = "direct effects of L7H8")
# %%
def show_batch_result(batch, start = 40, end = 47, per_head_direct_effect = per_head_direct_effect, all_layer_direct_effect = all_layer_direct_effect):
    print(model.to_string(all_owt_tokens[batch, 0:start]))
    print("...")
    print(model.to_string(all_owt_tokens[batch, start:end]))
    print("...")
    print(model.to_string(all_owt_tokens[batch, end:]))
    show_input(per_head_direct_effect[..., batch, start:end].mean(-1),all_layer_direct_effect[:, batch, start:end].mean(-1), title = f"Direct Effect of Heads on batch {batch}")



def create_scatter_of_backup_of_component(heads = None, mlp_layers = None, return_slope = False, return_CRE = False):
    """"
    this function:
    1) gets the direct effect of all a component when sample ablating it
    2) gets the total accumulated backup of the component for each prompt and position
    3) plots the clean direct effect vs accumulated backup

    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    return_CRE: bad coding practic eto wherei  just return cre if i'm lazy
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
    
    if return_CRE:
        return total_backup
    
    #  3) plots the clean direct effect vs accumulated backup

    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else all_layer_direct_effect[layer].flatten().cpu()
    assert direct_effects.shape == total_backup.flatten().cpu().shape
    if not return_slope:
        fig = go.Figure()
        text_labels = [f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]

        scatter_plot = go.Scatter(
            x = direct_effects,
            y = total_backup.flatten().cpu(),
            text=text_labels,  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Compensatory Response Size"
        )
        fig.add_trace(scatter_plot)

        second_scatter = go.Scatter(
            x = direct_effects,
            y = head_backup.flatten().cpu() ,
            text=text_labels,  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Response Size of just Attention Blocks"
        )
        fig.add_trace(second_scatter)

        third_scatter = go.Scatter(
            x = direct_effects,
            y = mlp_backup.flatten().cpu() ,
            text=text_labels, # Set the hover labels to the text attribute
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
            x=torch.linspace(-1 * max_x,max_x,100),
            y=torch.linspace(-1 * max_x,max_x,100) * slope + intercept,
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
if in_notebook_mode:
    #create_scatter_of_backup_of_component(mlp_layers = [10])
    #create_scatter_of_backup_of_component(heads = [(10,7)])
    create_scatter_of_backup_of_component(heads = [(1,8)]) # the head in gpt2-small which has insane downstream impact
    ablated_de, ablated_layer_de = dir_effects_from_sample_ablating(attention_heads=[(1,8)])
    #show_batch_result(27, start = 17, end = 24, per_head_direct_effect = (ablated_de - per_head_direct_effect), all_layer_direct_effect = (ablated_layer_de - all_layer_direct_effect))
# %%

slopes_of_head_backup = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
cre_of_heads = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        slopes_of_head_backup[layer, head] = create_scatter_of_backup_of_component(heads = [(layer, head)], return_slope = True)
        cre_of_heads[layer, head] = create_scatter_of_backup_of_component(heads = [(layer, head)], return_CRE = True).mean((-2,-1))


fig = imshow(slopes_of_head_backup, title = f"Slopes of Head Backup in {model_name}",
       text_auto = True, width = 800, height = 800, return_fig=True) # show a number above each square)
if in_notebook_mode: 
    fig.show()

# %%

# # save figure
# fig.write_image(f"slopes_of_head_backup_{safe_model_name}.png")



# # %%
# # create another figure plotting the cre vs the direct effect of the head
# # Flatten the tensors for plotting
# flattened_avg_direct_effect = per_head_direct_effect.mean((-1,-2)).flatten().cpu().numpy()
# layers_list = [l for l in range(model.cfg.n_layers) for _ in range(model.cfg.n_heads)]

# fig2 = go.Figure()
# fig2.add_trace(go.Scatter(
#     x=flattened_avg_direct_effect,
#     y=cre_of_heads.flatten(),
#     mode='markers',
#     marker=dict(
#         size=10,
#         opacity=0.5,
#         line=dict(width=1),
#         color=layers_list,  # Setting color based on layer
#         colorscale='Viridis',  # Using Viridis colorscale, but you can choose any other available colorscale
#         colorbar=dict(title='Layer'),  # Adding a colorbar to denote layers
#     ),
#     text=[f"Layer {l}, Head {h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
# ))

# # Add titles and labels
# fig2.update_layout(
#     title=f"Compensatory Response Effect vs Average Direct Effect in {model_name}",
#     xaxis_title="Average Direct Effect",
#     yaxis_title="Average Compensatory Response Effect",
#     hovermode="closest"  # Hover over data points to see the text
# )
# if in_notebook_mode:
#     fig2.show()


# # Save the figure
# fig2.write_image(f"cre_vs_avg_direct_effect_{safe_model_name}.png")

# %%
# Do the same thing but for MLP layers
# slopes_of_mlp_backup = torch.zeros((model.cfg.n_layers))
# for layer in tqdm(range(model.cfg.n_layers)):
#     slopes_of_mlp_backup[layer] = create_scatter_of_backup_of_component(mlp_layers = [layer], return_slope = True)

# if in_notebook_mode: 
#     fig.show()
# %%
# fig = imshow(einops.repeat(slopes_of_mlp_backup, "a -> a 1"), title = f"Slopes of MLP Backup in {model_name}",
#        text_auto = True, width = 800, height = 800, return_fig=True)# show a number above each square)
# fig.write_image(f"slopes_of_mlp_backup_{safe_model_name}.png")
# %%

if in_notebook_mode:
    layer = 5
    head = 11
    temp_head_effect, temp_mlp_effect = dir_effects_from_sample_ablating(attention_heads = [(layer, head)])
    show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)),
            title = f"Logit Diff Diff of Downstream Components upon ablation of {layer}.{head}")

    layer = 10
    temp_head_effect, temp_mlp_effect = dir_effects_from_sample_ablating(mlp_layers = [10])
    show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)),
            title = f"Logit Diff Diff of Downstream Components upon ablation of layer {layer}")
    
    #hist(all_layer_direct_effect[-1].flatten().cpu())

# %%





# %% 
# Get the extent to which heads can have downstream heads act positively, rather than negatively
# pairs_of_head_backup = {}
# pairs_of_mlp_backup = {}
# threshold = 0.05
# for layer in tqdm(range(model.cfg.n_layers)):
#     for head in range(model.cfg.n_heads):
#         temp_head_effect, temp_mlp_effect = dir_effects_from_sample_ablating(attention_heads = [(layer, head)])
#         head_backup: Float[Tensor, "layer head"] = temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2))
#         mlp_backup: Float[Tensor, "layer"] = temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2))

#         if (head_backup > threshold).sum() > 0:
#             pairs_of_head_backup[(layer, head)] = True
#         else:
#             pairs_of_head_backup[(layer, head)] = False

#         if (mlp_backup > threshold).sum() > 0:
#             pairs_of_mlp_backup[layer] = True
#         else:
#             pairs_of_mlp_backup[layer] = False

# for layer, head in pairs_of_head_backup.keys():
#     if pairs_of_head_backup[(layer, head)]:
#         print((layer, head))

# for layer in range(12):
#     if pairs_of_mlp_backup[layer]:
#         print(layer)

# %%

def create_scatter_of_change_from_component(heads = None, mlp_layers = None, return_slope = False):
    """"
    this function:
    1) gets the direct effect of all a component when sample ablating it
    2) gets the CHANGE IN LOGIT CONTRIBUTION for each prompt and position
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
    
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect, ablated_cache = dir_effects_from_sample_ablating(attention_heads=heads, mlp_layers=mlp_layers, return_cache = True)


    # get the final direct effects under ablation
    final_resid_stream: Float[Tensor, "batch pos d_model"] = ablated_cache[f'blocks.{model.cfg.n_layers - 1}.hook_resid_post']
    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(owt_tokens)
    final_direct_effect: Float[Tensor, "batch pos_minus_one"] = residual_stack_to_direct_effect(final_resid_stream, token_residual_directions, True)

    # get change in direct effect compared to original
    change_in_direct_effect = (final_direct_effect - last_layer_direct_effect) # if you wanna compare to just CRE - (ablated_per_head_direct_effect[layer, head] - per_head_direct_effect[layer, head])
    
    #  3) plots the clean direct effect vs accumulated backup
    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else all_layer_direct_effect[layer].flatten().cpu()
    assert direct_effects.shape == change_in_direct_effect.flatten().cpu().shape

    # get a best fit line
    slope, intercept = np.linalg.lstsq(np.vstack([direct_effects, np.ones(len(direct_effects))]).T, change_in_direct_effect.flatten().cpu(), rcond=None)[0]

    if not return_slope:
        fig = go.Figure()
        text_labels = [f"Batch {i[0]}, Pos {i[1]}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]


        scatter_plot = go.Scatter(
            x = direct_effects,
            y = change_in_direct_effect.flatten().cpu(),
            text=text_labels,  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Change in Final Logits vs Direct Effect"
        )
        fig.add_trace(scatter_plot)
        max_x = max(direct_effects.abs())

        # add line of best fit
        fig.add_trace(go.Scatter(
            x=torch.linspace(-1 * max_x,max_x,100),
            y=torch.linspace(-1 * max_x,max_x,100) * slope + intercept,
            mode='lines',
            name='lines'
        ))

        # add dashed y=-x line
        fig.add_trace(go.Scatter(
            x=torch.linspace(-1 * max_x,max_x,100),
            y=torch.linspace(-1 * max_x,max_x,100) * -1,
            mode='lines',
            name='y = -x',
            line = dict(dash = 'dash')
        ))

        component = heads if heads is not None else mlp_layers
        fig.update_layout(
            title=f"Change in Final Logits vs Direct Effect for {component} in {model_name} for each Position and Batch" if heads is not None 
            else f"Change in Final Logits vs Direct Effect for MLP Layer {component} in {model_name} for each Position and Batch",
        )
        fig.update_xaxes(title = f"Direct Effect of Head {heads[0]}" if heads is not None else f"Direct Effect of MLP Layer {mlp_layers[0]}")
        fig.update_yaxes(title = "Change in Final Logits")
        fig.update_layout(width=900, height=500)
        fig.show()
    
    if return_slope:
        return slope
# %%
#create_scatter_of_change_from_component(heads = [(9,6)])
# %%


def get_threholded_de_cre(heads = None, mlp_layers = None, thresholds = [0.5]):
    """"
    this function calculates direct effect of component on clean and sample ablation,
    and then returns an averaged direct effect and compensatory response effect of the component
    of 'significant' tokens, defined as tokens with a direct effect of at least threshold 


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
    

    assert downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() == 0
    if heads is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer):].sum(0)
        total_backup = head_backup + mlp_backup
    if mlp_layers is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer+1):].sum(0)
        total_backup = head_backup + mlp_backup
    

    # 3) filter for indices wherer per_head_direct_effect is greater than threshold
    to_return = {}
    for threshold in thresholds:
        mask_direct_effect: Float[Tensor, "batch pos"] = per_head_direct_effect[layer, head] > threshold
        # Using masked_select to get the relevant values based on the mask.
        selected_cre = total_backup.masked_select(mask_direct_effect)
        selected_de = per_head_direct_effect[layer, head].masked_select(mask_direct_effect)
        
    
        to_return[f"de_{threshold}"] = selected_de.mean().item()
        to_return[f"cre_{threshold}"] = selected_cre.mean().item()
        to_return[f"num_thresholded_{threshold}"] = selected_cre.shape[0]
        
    return to_return



def plot_thresholded_de_vs_cre(thresholds = [1]):
    thresholded_de = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))
    thresholded_cre = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))
    thresholded_count = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))

    for layer in tqdm(range(model.cfg.n_layers)):
        for head in range(model.cfg.n_heads):
            dict_results = get_threholded_de_cre(heads = [(layer, head)], thresholds = thresholds)
            for i, threshold in enumerate(thresholds):
                thresholded_de[i, layer, head] = dict_results[f"de_{threshold}"]
                thresholded_cre[i, layer, head] = dict_results[f"cre_{threshold}"]
                thresholded_count[i, layer, head] = dict_results[f"num_thresholded_{threshold}"]

    
    layers_list = [l for l in range(model.cfg.n_layers) for _ in range(model.cfg.n_heads)]
    
    fig = go.Figure()

    for i, threshold in enumerate(thresholds):
        visible = (i == 0)  # Only the first threshold is visible initially
        
        scatter_trace = go.Scatter(
            x=thresholded_de[i].flatten(),
            y=thresholded_cre[i].flatten(),
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.5,
                line=dict(width=1),
                color=layers_list,
                colorscale='Viridis',
                colorbar=dict(title='Layer', y=10),
            ),
            text=[f"Layer {l}, Head {h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
            visible=visible,
            name=f"Threshold: {threshold}"
        )

        line_trace = go.Scatter(
            x=torch.linspace(0, max(thresholded_de[i].masked_fill(torch.isnan(thresholded_de[i]), -float('inf')).flatten()), 100),
            y=torch.linspace(0, max(thresholded_de[i].masked_fill(torch.isnan(thresholded_de[i]), -float('inf')).flatten()), 100),
            mode='lines',
            line=dict(dash='dash'),
            visible=visible
        )
        
        fig.add_trace(scatter_trace)
        fig.add_trace(line_trace)
    
    frames = []
    for i, threshold in enumerate(thresholds):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=thresholded_de[i].flatten().numpy(),
                    y=thresholded_cre[i].flatten().numpy(),
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.5,
                        line=dict(width=1),
                        color=layers_list,
                        colorscale='Viridis',
                        colorbar=dict(title='Layer', y=10),
                    ),
                    text=[f"Layer {l}, Head {h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
                )
            ],
            name=str(threshold)   # Use the threshold as the name of the frame
        )
        frames.append(frame)

    
    fig.frames = frames

    
    
    steps = []
    for i, threshold in enumerate(thresholds):
        step = {
            'args': [
                [str(threshold)],  # Frame name to show
                {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}
            ],
            'label': str(threshold),
            'method': 'animate'
        }
        steps.append(step)


    sliders = [dict(
        yanchor='top',
        xanchor='left',
        currentvalue={'font': {'size': 16}, 'prefix': 'Threshold: ', 'visible': True, 'xanchor': 'right'},
        transition={'duration': 300, 'easing': 'cubic-in-out'},
        pad={'b': 10, 't': 50},
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=f"Thresholded Compensatory Response Effect vs Average Direct Effect in {model_name}",
        xaxis_title="Average Direct Effect",
        yaxis_title="Average Compensatory Response Effect",
        hovermode="closest",
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        width=1000,
        height=400,
        showlegend=False,
    )
    
    if in_notebook_mode:
        fig.show()
    
    fig.write_html(f"threshold_figures/threshold_{safe_model_name}_cre_vs_avg_direct_effect_slider.html")
# %%
#plot_thresholded_de_vs_cre([-99999999999999999])
plot_thresholded_de_vs_cre([0.15 * i for i in range(0,10)])
# %%
