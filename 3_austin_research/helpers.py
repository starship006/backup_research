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
def dir_effects_from_sample_ablating(model = None, clean_tokens: Float[Tensor, "batch seq_len"] = None, corrupted_tokens: Float[Tensor, "batch seq_len"] = None, attention_heads = None, mlp_layers = None, neurons = None, return_cache = False) -> Union[ActivationCache, Tuple[Float[Tensor, "heads batch pos_minus_one"], Float[Tensor, "n_layer batch pos_minus_one"]]]:
    """this function gets the new direct effect of all the heads when sample ablating a component
    it uses the global cache, owt_tokens, corrupted_owt_tokens

    attention_heads: list of tuples of (layer, head) to ablate
    clean_tokens: [batch, seq_len] tensor of clean tokens
    mlp_layers: list of layers to ablate
    neurons: list of tuples of (layer, neuron) to ablate
    return_cache: whether to return the cache as well as the direct effect
    """

    if model is None or clean_tokens is None or corrupted_tokens is None:
        raise ValueError("model, clean_tokens, and corrupted_tokens cannot be None")

    # don't accept if more than one input is none
    assert sum([attention_heads is not None, mlp_layers is not None, neurons is not None]) == 1
    
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
    clean_tokens: Float[Tensor, "batch 1"] = None,
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    smaller_logits = logits[:, :-1, :]
    smaller_correct = clean_tokens[:, 1:].unsqueeze(-1)
    answer_logits: Float[Tensor, "batch 2"] = smaller_logits.gather(dim=-1, index=smaller_correct)
    return answer_logits.squeeze() # get rid of last index of size one