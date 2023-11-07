from imports import *
import inspect

# this is a decorator for functions to be partialized
def to_partial(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    wrapper.to_partial = True
    return wrapper



# this is the function which takes all of these possibly partial functions, and puts in shared args into them.
def return_partial_functions(**shared_kwargs):
    partials = {}
    for name, func in globals().items():
        if callable(func) and hasattr(func, 'to_partial') and func.to_partial:
            # Get the names of the function's arguments
            arg_names = set(inspect.signature(func).parameters.keys())
            
            # Filter shared_kwargs to only include keys that are in the function's arguments
            applicable_kwargs = {k: v for k, v in shared_kwargs.items() if k in arg_names}
            
            partials[name] = partial(func, **applicable_kwargs)
    return partials




@to_partial
def return_item(item):
    return item

@to_partial
def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()

@to_partial
def residual_stack_to_direct_effect(
    residual_stack: Float[Tensor, "... batch pos d_model"],
    effect_directions: Float[Tensor, "batch pos d_model"],
    apply_last_ln = True,
    scaling_cache = None
) -> Float[Tensor, "... batch pos_mins_one"]:
    '''
    Gets the direct effect towards a direction for a given stack of components in the residual stream.
    NOTE: IGNORES THE VERY LAST PREDICTION AND FIRST CLEAN TOKEN; WE DON'T KNOW THE ACTUAL PREDICTED ANSWER FOR IT!

    residual_stack: [... batch pos d_model] components of d_model vectors to measure direct effect from
    effect_directions: [batch pos d_model] vectors in d_model space that correspond to 'direct effect'
    scaling_cache = the cache to use for the scaling; defaults to the global clean cache
    '''

    if scaling_cache is None:
        raise ValueError("scaling_cache cannot be None")
    
    scaled_residual_stack = scaling_cache.apply_ln_to_stack(residual_stack, layer=-1, has_batch_dim=True) if apply_last_ln else residual_stack
    
    # remove the last prediction, and the direction of the zeroth token
    scaled_residual_stack = scaled_residual_stack[..., :, :-1, :]
    effect_directions = effect_directions[:, 1:, :]

    return einops.einsum(scaled_residual_stack, effect_directions, "... batch pos d_model, batch pos d_model -> ... batch pos")


# test to ensure functions work: compare the predicted direct effect from the residual stack function to the actual direct effect

# last_layer_direct_effect: Float[Tensor, "batch pos_minus_one"] = residual_stack_to_direct_effect(
#     cache[f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"],
#     model.tokens_to_residual_directions(owt_tokens),
#     )

# first_batch_with_bias = [predicted_last_layer_direct_effect[0, i] + model.b_U[owt_tokens[0, i + 1].item()] for i in range(49)]
# # compare the predicted 'direct effect' to the logits on the correct token. these should be the same
# key_logits = [logits[0, index, token].item() for index, token in enumerate(owt_tokens[0, 1:])]

# # compare the predicted 'direct effect' to the logits on the correct token. these should be the same
# for i in range(10):
#     print(first_batch_with_bias[i], key_logits[i])



@to_partial
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


@to_partial
def collect_direct_effect(de_cache: ActivationCache = None, correct_tokens: Float[Tensor, "batch seq_len"] = None, model: HookedTransformer = None, 
                           title = "Direct Effect of Heads", display = True, collect_individual_neurons = False) -> Tuple[Float[Tensor, "n_layer n_head batch pos_minus_one"], Float[Tensor, "n_layer batch pos_minus_one"], Float[Tensor, "n_layer d_mlp batch pos_minus_one"]]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head and neuron on each token.
    
    returns tuple of tensors of per-head, per-mlp-layer, per-neuron of direct effects

    cache: cache of activations from the model
    correct_tokens: [batch, seq_len] tensor of correct tokens
    title: title of the plot (relavant if display == True)
    display: whether to display the plot or return the data; if False, returns [head, pos] tensor of direct effects
    """

    if de_cache is None or correct_tokens is None or model is None:
        raise ValueError("de_cache, correct_tokens, and model must not be None")

    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(correct_tokens)
    
    # get the direct effect of heads by positions
    clean_per_head_residual: Float[Tensor, "head batch seq d_model"] = de_cache.stack_head_results(layer = -1, return_labels = False, apply_ln = False)
    
    #print(clean_per_head_residual.shape)
    per_head_direct_effect: Float[Tensor, "heads batch pos_minus_one"] = residual_stack_to_direct_effect(clean_per_head_residual, token_residual_directions, True, scaling_cache = de_cache)
    
    
    per_head_direct_effect = einops.rearrange(per_head_direct_effect, "(n_layer n_head) batch pos -> n_layer n_head batch pos", n_layer = model.cfg.n_layers, n_head = model.cfg.n_heads)
    #assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, tokens.shape[0], tokens.shape[1])

    # get the outputs of the neurons
    direct_effect_mlp: Float[Tensor, "n_layer d_mlp batch pos_minus_one"] = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp, correct_tokens.shape[0], correct_tokens.shape[1] - 1))
    
    # iterate over every neuron to avoid memory issues
    if collect_individual_neurons:
        for neuron in tqdm(range(model.cfg.d_mlp)):
            single_neuron_output: Float[Tensor, "n_layer batch pos d_model"] = de_cache.stack_neuron_results(layer = -1, neuron_slice = (neuron, neuron + 1), return_labels = False, apply_ln = False)
            direct_effect_mlp[:, neuron, :, :] = residual_stack_to_direct_effect(single_neuron_output, token_residual_directions, scaling_cache = de_cache).cpu()
    # get per mlp layer effect
    all_layer_output: Float[Tensor, "n_layer batch pos d_model"] = torch.zeros((model.cfg.n_layers, correct_tokens.shape[0], correct_tokens.shape[1], model.cfg.d_model)).cuda()
    for layer in range(model.cfg.n_layers):
        all_layer_output[layer, ...] = de_cache[f'blocks.{layer}.hook_mlp_out']

    all_layer_direct_effect: Float["n_layer batch pos_minus_one"] = residual_stack_to_direct_effect(all_layer_output, token_residual_directions, scaling_cache = de_cache).cpu()


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
    


@to_partial
def dir_effects_from_sample_ablating(model = None, clean_tokens: Float[Tensor, "batch seq_len"] = None, corrupted_tokens: Float[Tensor, "batch seq_len"] = None, attention_heads = None,
 mlp_layers = None, neurons = None, return_cache = False, zero_ablate = False) -> Union[ActivationCache, Tuple[Float[Tensor, "heads batch pos_minus_one"], Float[Tensor, "n_layer batch pos_minus_one"]]]:
    """this function gets the new direct effect of all the heads when sample ablating a component
    it uses the global cache, owt_tokens, corrupted_owt_tokens

    attention_heads: list of tuples of (layer, head) to ablate
    clean_tokens: [batch, seq_len] tensor of clean tokens
    mlp_layers: list of layers to ablate
    neurons: list of tuples of (layer, neuron) to ablate
    return_cache: whether to return the cache as well as the direct effect
    zero_ablate: whether to zero ablate instead of sample ablating
    """

    if model is None or clean_tokens is None or corrupted_tokens is None:
        raise ValueError("model, clean_tokens, and corrupted_tokens cannot be None")

    # don't accept if more than one input is none
    assert sum([attention_heads is not None, mlp_layers is not None, neurons is not None]) == 1
    

    if zero_ablate:
        if attention_heads is not None:
            new_cache: ActivationCache = act_patch(model, clean_tokens, [Node("z", layer, head) for (layer,head) in attention_heads],
                                return_item, new_cache = "zero", apply_metric_to_cache= True)
        elif mlp_layers is not None:
            new_cache: ActivationCache = act_patch(model, clean_tokens, [Node("mlp_out", layer) for layer in mlp_layers],
                                return_item, new_cache = "zero", apply_metric_to_cache= True)
        elif neurons is not None:
            new_cache: ActivationCache = act_patch(model, clean_tokens, [Node("post", layer = layer, neuron = neuron) for (layer,neuron) in neurons],
                                return_item, new_cache = "zero", apply_metric_to_cache= True)
        else:
            raise ValueError("Must specify attention_heads, mlp_layers, or neurons")
    else:
        if attention_heads is not None:
            new_cache: ActivationCache = act_patch(model, clean_tokens, [Node("z", layer, head) for (layer,head) in attention_heads],
                                return_item, corrupted_tokens, apply_metric_to_cache= True)
        elif mlp_layers is not None:
            new_cache: ActivationCache = act_patch(model, clean_tokens, [Node("mlp_out", layer) for layer in mlp_layers],
                                return_item, corrupted_tokens, apply_metric_to_cache= True)
        elif neurons is not None:
            new_cache: ActivationCache = act_patch(model, clean_tokens, [Node("post", layer = layer, neuron = neuron) for (layer,neuron) in neurons],
                                return_item, corrupted_tokens, apply_metric_to_cache= True)
        else:
            raise ValueError("Must specify attention_heads, mlp_layers, or neurons")

    head_direct_effect, mlp_layer_direct_effect = collect_direct_effect(new_cache, clean_tokens, model, display = False, collect_individual_neurons = False)
                                            
    if return_cache:
        return head_direct_effect, mlp_layer_direct_effect, new_cache
    else:
        return head_direct_effect, mlp_layer_direct_effect
    

@to_partial
def get_correct_logit_score(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_tokens: Float[Tensor, "batch seq"] = None,
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    TESTED
    '''
    smaller_logits = logits[:, :-1, :]
    smaller_correct = clean_tokens[:, 1:].unsqueeze(-1)
    answer_logits: Float[Tensor, "batch 2"] = smaller_logits.gather(dim=-1, index=smaller_correct)
    return answer_logits.squeeze() # get rid of last index of size one

# test with this
# a = torch.tensor([[[0,1,2,3,4], [10,11,12,13,14], [100,101,120,103,140]], 
#                   [[10,999,2,3,4], [110,191,120,13,14], [1100,105,120,103,140]]])
# get_correct_logit_score(a, clean_tokens = torch.tensor([[3, 2, 4], [0,1,2]]))

def print_tokens(model, all_owt_tokens, batch, start = 40, end = 47):
    """
    Prints the tokens for a batch. Shares same indexing.
    """
    print(model.to_string(all_owt_tokens[batch, 0:start]))
    print("...")
    print(model.to_string(all_owt_tokens[batch, start:end]))
    # print("...")
    # print(model.to_string(all_owt_tokens[batch, end:]))
def change_in_ld_calc(logits, heads = None, mlp_layers = None, num_runs = 5):
    """
    runs activation patching over component and returns new avg_correct_logit_score, averaged over num_runs runs
    this is the average logit of the correct token upon num_runs sample ablations
    logits just used for size
    """
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    # use many shuffled!
    logits_accumulator = torch.zeros_like(logits)
    for i in range(num_runs):
        # Shuffle owt_tokens by batch
        shuffled_corrupted_owt_tokens = shuffle_owt_tokens_by_batch(corrupted_owt_tokens)
        # Calculate new_logits using act_patch
        new_logits = act_patch(model, owt_tokens, nodes, return_item, shuffled_corrupted_owt_tokens, apply_metric_to_cache=False)
        logits_accumulator += new_logits

    avg_logits = logits_accumulator / num_runs
    # get change in direct effect compared to original
    avg_correct_logit_score = get_correct_logit_score(avg_logits)
    return avg_correct_logit_score
def show_batch_result(model, all_owt_tokens, batch,  per_head_direct_effect, all_layer_direct_effect, start = 40, end = 47):
    """
    highlights the text selection, along with the mean effect of the range
    indexed similariy to python, where start is inclusive and end is exclusive 

    recall that the per_head_direct_effect is one length shorter than the input, since it doesn't have the first token
    so, if the interesting self-repair you are observing seems to be at pos 12, this means it is for the prediction of token 13
    """
    
    print_tokens(model, all_owt_tokens, batch, start, end)
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
    
    if(downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() != 0):
        print("expect assymetry since different LNs used")
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
        
        text_labels = [f"Batch {i[0]}, Pos {i[1]}: {model.to_string(all_owt_tokens[i[0], i[1]:(i[1] + 1)])} --> {model.to_string(all_owt_tokens[i[0], (i[1] + 1):(i[1] + 2)])}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]

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
def shuffle_owt_tokens_by_batch(owt_tokens):
    batch_size, num_tokens = owt_tokens.shape
    shuffled_owt_tokens = torch.zeros_like(owt_tokens)
    
    for i in range(batch_size):
        perm = torch.randperm(num_tokens)
        shuffled_owt_tokens[i] = owt_tokens[i, perm]
        
    return shuffled_owt_tokens
def create_scatter_of_change_from_component(model, logits, owt_tokens, new_logits_after_sample_wrapper, per_head_direct_effect, all_layer_direct_effect, all_owt_tokens, heads = None, mlp_layers = None, return_slope = False, zero_ablate = False, force_through_origin = False, num_runs = 1):
    """"
    this function:
    1) gets the direct effect of all a component when sample ablating it
    2) gets the CHANGE IN LOGIT CONTRIBUTION for each prompt and position
    3) plots the clean direct effect vs accumulated backup

    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    force_through_origin: if true, will force the line of best fit to go through the origin
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
    
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    if zero_ablate:
        new_logits = act_patch(model, owt_tokens, nodes, return_item, new_cache="zero", apply_metric_to_cache= False)
        diff_in_logits = new_logits - logits
        change_in_direct_effect = get_correct_logit_score(diff_in_logits)
    else:
        new_logits = new_logits_after_sample_wrapper(heads, mlp_layers, num_runs, logits)
        change_in_direct_effect = new_logits - get_correct_logit_score(logits)
    
    #  3) plots the clean direct effect vs accumulated backup
    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else all_layer_direct_effect[layer].flatten().cpu()
    assert direct_effects.shape == change_in_direct_effect.flatten().cpu().shape

    # get a best fit line
    if force_through_origin:
        slope = np.linalg.lstsq(direct_effects.reshape(-1, 1), change_in_direct_effect.flatten().cpu(), rcond=None)[0][0]
        intercept = 0
    else:
        slope, intercept = np.linalg.lstsq(np.vstack([direct_effects, np.ones(len(direct_effects))]).T, change_in_direct_effect.flatten().cpu(), rcond=None)[0]
    
    if not return_slope:
        fig = go.Figure()
        text_labels = [f"Batch {i[0]}, Pos {i[1]}: {model.to_string(all_owt_tokens[i[0], i[1]:(i[1] + 1)])} --> {model.to_string(all_owt_tokens[i[0], (i[1] + 1):(i[1] + 2)])}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]


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
            title=f"Change in Final Logits vs Direct Effect for {component} in {model_name} for each Position and Batch. zero_ablate = {zero_ablate}" if heads is not None 
            else f"Change in Final Logits vs Direct Effect for MLP Layer {component} in {model_name} for each Position and Batch. zero_ablate = {zero_ablate}",
        )
        fig.update_xaxes(title = f"Direct Effect of Head {heads[0]}" if heads is not None else f"Direct Effect of MLP Layer {mlp_layers[0]}")
        fig.update_yaxes(title = "Change in Final Logits")
        fig.update_layout(width=900, height=500)
        fig.show()
    
    if return_slope:
        return slope
def get_threshold_from_percent(logits, threshold_percent_filter):
    logits_flat = logits.flatten()
    sorted_logits = logits_flat.sort()[0]
    index = int((1 - threshold_percent_filter) * sorted_logits.size(0))
    threshold_value = sorted_logits[index]
    return threshold_value
def get_top_self_repair_prompts(logits, heads = None, mlp_layers = None, topk = 10, num_runs = 5, logit_diff_self_repair = True, threshold_percent_filter = 1):
    """
    if logit_diff_self_repair, Top self repair is calcualted by seeing how little the logits change.
    if not, it just calculated by whichever prompts, when ablating the component, changes the most positively in logits.

    threshold_filter controls for which examples to consider; if not zero, it will only consider examples where the absolute direct effect of the component is at least threshold_filter
    """
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        direct_effect = per_head_direct_effect[layer, head]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
        direct_effect = all_layer_direct_effect[layer]
    else:
        raise Exception("No heads or mlp layers given")
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    # get change in direct effect compared to original
    change_in_direct_effect = new_logits_after_sample_wrapper(heads, mlp_layers, num_runs, logits) - get_correct_logit_score(logits)
    
    # get topk
    threshold_filter = get_threshold_from_percent(direct_effect.abs(), threshold_percent_filter)
    mask_direct_effect: Float[Tensor, "batch pos"] = direct_effect.abs() > threshold_filter
    
    if logit_diff_self_repair:
        # Using masked_select to get the relevant values based on the mask.
        change_in_direct_effect = change_in_direct_effect.masked_fill(~mask_direct_effect, 99999)
        topk_indices = topk_of_Nd_tensor(-1 * change_in_direct_effect.abs(), k = topk) # -1 cause we want the minimim change in logits
    else:
        change_in_direct_effect = change_in_direct_effect.masked_fill(~mask_direct_effect, -99999)
        topk_indices = topk_of_Nd_tensor(change_in_direct_effect, k = topk)
    return topk_indices
