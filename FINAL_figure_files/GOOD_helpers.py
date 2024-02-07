from imports import *

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


def collect_direct_effect(de_cache: ActivationCache = None, correct_tokens: Float[Tensor, "batch seq_len"] = None, model: HookedTransformer = None, 
                           title = "Direct Effect of Heads", display = True, collect_individual_neurons = False, cache_for_scaling: ActivationCache = None) -> Tuple[Float[Tensor, "n_layer n_head batch pos_minus_one"], Float[Tensor, "n_layer batch pos_minus_one"]]:#, Float[Tensor, "n_layer d_mlp batch pos_minus_one"]]:
    """
    Given a cache of activations, and a set of correct tokens, returns the direct effect of each head and neuron on each token.
    
    returns tuple of tensors of per-head, per-mlp-layer, per-neuron of direct effects

    cache: cache of activations from the model
    correct_tokens: [batch, seq_len] tensor of correct tokens
    title: title of the plot (relavant if display == True)
    display: whether to display the plot or return the data; if False, returns [head, pos] tensor of direct effects
    cache_for_scaling: the cache to use for the scaling; defaults to the global clean cache
    """

    device = "cuda" if correct_tokens.get_device() >= 0 else "cpu"

    if de_cache is None or correct_tokens is None or model is None:
        raise ValueError("de_cache, correct_tokens, and model must not be None")

    if cache_for_scaling is None:
        cache_for_scaling = de_cache

    token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(correct_tokens)
    
    # get the direct effect of heads by positions
    clean_per_head_residual: Float[Tensor, "head batch seq d_model"] = de_cache.stack_head_results(layer = -1, return_labels = False, apply_ln = False)
    
    #print(clean_per_head_residual.shape)
    per_head_direct_effect: Float[Tensor, "heads batch pos_minus_one"] = residual_stack_to_direct_effect(clean_per_head_residual, token_residual_directions, True, scaling_cache = cache_for_scaling)
    
    
    per_head_direct_effect = einops.rearrange(per_head_direct_effect, "(n_layer n_head) batch pos -> n_layer n_head batch pos", n_layer = model.cfg.n_layers, n_head = model.cfg.n_heads)
    #assert per_head_direct_effect.shape == (model.cfg.n_heads * model.cfg.n_layers, tokens.shape[0], tokens.shape[1])

    # get the outputs of the neurons
    direct_effect_mlp: Float[Tensor, "n_layer d_mlp batch pos_minus_one"] = torch.zeros((model.cfg.n_layers, model.cfg.d_mlp, correct_tokens.shape[0], correct_tokens.shape[1] - 1))
    
    # iterate over every neuron to avoid memory issues
    if collect_individual_neurons:
        for neuron in range(model.cfg.d_mlp):
            single_neuron_output: Float[Tensor, "n_layer batch pos d_model"] = de_cache.stack_neuron_results(layer = -1, neuron_slice = (neuron, neuron + 1), return_labels = False, apply_ln = False)
            direct_effect_mlp[:, neuron, :, :] = residual_stack_to_direct_effect(single_neuron_output, token_residual_directions, scaling_cache = cache_for_scaling)
    # get per mlp layer effect
    all_layer_output: Float[Tensor, "n_layer batch pos d_model"] = torch.zeros((model.cfg.n_layers, correct_tokens.shape[0], correct_tokens.shape[1], model.cfg.d_model)).to(device)
    for layer in range(model.cfg.n_layers):
        all_layer_output[layer, ...] = de_cache[f'blocks.{layer}.hook_mlp_out']

    all_layer_direct_effect: Float["n_layer batch pos_minus_one"] = residual_stack_to_direct_effect(all_layer_output, token_residual_directions, scaling_cache = cache_for_scaling)


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


# def print_tokens(model, all_owt_tokens, batch, start = 40, end = 47):
#     """
#     Prints the tokens out of dataset given a batch and position index. Shares same indexing. Printing starts from beginning
#     Start: where to begin section 2 
#     End: where to end section 2
#     """
#     print(model.to_string(all_owt_tokens[batch, 0:start]))
#     print("...")
#     print(model.to_string(all_owt_tokens[batch, start:end]))

def get_correct_logit_score(
    logits: Float[Tensor, "batch seq d_vocab"],
    clean_tokens: Float[Tensor, "batch seq"],
):
    '''
    Returns the logit of the next token

    If per_prompt=True, return the array of differences rather than the average.
    '''
    smaller_logits = logits[:, :-1, :]
    smaller_correct = clean_tokens[:, 1:].unsqueeze(-1)
    answer_logits: Float[Tensor, "batch 2"] = smaller_logits.gather(dim=-1, index=smaller_correct)
    return answer_logits.squeeze() # get rid of last index of size one

# test with this
# a = torch.tensor([[[0,1,2,3,4], [10,11,12,13,14], [100,101,120,103,140]], 
#                   [[10,999,2,3,4], [110,191,120,13,14], [1100,105,120,103,140]]])
# get_correct_logit_score(a, clean_tokens = torch.tensor([[3, 2, 4], [0,1,2]]))


def shuffle_owt_tokens_by_batch(owt_tokens: torch.Tensor, offset_shuffle=2) -> torch.Tensor:
    """Shuffles the prompts in a batch by just moving them by an offset."""
    # Roll the batch dimension by the specified offset
    if offset_shuffle == 0:
        print("Warning: offset_shuffle = 0, so no shuffling is happening")
    
    shuffled_owt_tokens = torch.roll(owt_tokens, shifts=offset_shuffle, dims=0)
    return shuffled_owt_tokens


def return_item(item):
    return item

def create_layered_scatter(
    heads_x: Float[Tensor, "layer head"],
    heads_y: Float[Tensor, "layer head"], 
    model,
    x_title: str, 
    y_title: str, 
    plot_title: str,
    mlp_x: Union[Float[Tensor, "layer"], None] = None,
    mlp_y: Union[Float[Tensor, "layer"], None] = None
):
    """
    This function now also accepts x_data and y_data for MLP layers. 
    It plots properties of transformer heads and MLP layers with layered coloring and annotations.
    """
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    layer_colors = np.linspace(0, num_layers, num_layers, endpoint = False)
    
    # Annotations and colors for transformer heads
    head_annotations = [f"Layer {layer}, Head {head}" for layer, head in itertools.product(range(num_layers), range(num_heads))]
    head_marker_colors = [layer_colors[layer] for layer in range(num_layers) for _ in range(num_heads)]

    # Prepare MLP data if provided
    mlp_annotations = []
    mlp_marker_colors = []
    if mlp_x is not None and mlp_y is not None:
        mlp_annotations = [f"MLP Layer {layer}" for layer in range(num_layers)]
        mlp_marker_colors = [layer_colors[layer] for layer in range(num_layers)]
    # Flatten transformer heads data
    heads_x = heads_x.flatten().cpu().numpy() if heads_x.ndim > 1 else heads_x.cpu().numpy()
    heads_y = heads_y.flatten().cpu().numpy() if heads_y.ndim > 1 else heads_y.cpu().numpy()

    # Flatten MLP data if provided
    if mlp_x is not None and mlp_y is not None:
        mlp_x = mlp_x.flatten().cpu().numpy() if mlp_x.ndim > 1 else mlp_x.cpu().numpy()
        mlp_y = mlp_y.flatten().cpu().numpy() if mlp_y.ndim > 1 else mlp_y.cpu().numpy()

    # Create scatter plots
    scatter_heads = go.Scatter(
        x=heads_x,
        y=heads_y,
        text=head_annotations,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8,
            color=head_marker_colors,
            colorscale='Viridis',
            colorbar=dict(
                title='Layer',
                #tickvals=[0, num_layers - 1],
                #ticktext=[0, 1,2,1,1,1,1,1,1,1,1,1,1,11,1,1,3,4,5,5,num_layers - 1],
                orientation="h"
            ),
            line=dict(width=0.5, color='DarkSlateGrey')
        ),
        name="Attention Heads"
    )

    scatter_mlp = go.Scatter(
        x=mlp_x,
        y=mlp_y,
        text=mlp_annotations,
        mode='markers',
        name='MLP Layers',
        marker=dict(
            size=10,
            opacity=0.6,
            color=mlp_marker_colors,
            colorscale='Viridis',
            symbol='diamond',
            line=dict(width=1, color='Black')
        )
    ) if mlp_x is not None and mlp_y is not None else None

    # Create the figure and add the traces
    fig = go.Figure()
    fig.add_trace(scatter_heads)
    if scatter_mlp:
        fig.add_trace(scatter_mlp)

    # Update the layout
    fig.update_layout(
        title=f"{plot_title}",
        title_x=0.5,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="Component",
        width=900,
        height=500
    )

    return fig

def topk_of_Nd_tensor(tensor: Float[Tensor, "rows cols"], k: int):
    '''
    Helper function: does same as tensor.topk(k).indices, but works over 2D tensors.
    Returns a list of indices, i.e. shape [k, tensor.ndim].

    Example: if tensor is 2D array of values for each head in each layer, this will
    return a list of heads.
    '''
    i = torch.topk(tensor.flatten(), k).indices
    return np.array(np.unravel_index(utils.to_numpy(i), tensor.shape)).T.tolist()


def get_projection(from_vector: Float[Tensor, "batch d_model"], to_vector: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_model"]:
    assert from_vector.shape == to_vector.shape
    assert from_vector.ndim == 2
    
    dot_product = einops.einsum(from_vector, to_vector, "batch d_model, batch d_model -> batch")
    #length_of_from_vector = einops.einsum(from_vector, from_vector, "batch d_model, batch d_model -> batch")
    length_of_vector = einops.einsum(to_vector, to_vector, "batch d_model, batch d_model -> batch")
    

    projected_lengths = (dot_product) / (length_of_vector)
    projections = to_vector * einops.repeat(projected_lengths, "batch -> batch d_model", d_model = to_vector.shape[-1])
    return projections

def get_3d_projection(from_vector: Float[Tensor, "batch seq d_model"], to_vector: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
    assert from_vector.shape == to_vector.shape
    assert from_vector.ndim == 3
    
    dot_product = einops.einsum(from_vector, to_vector, "batch seq d_model, batch seq d_model -> batch seq")
    length_of_vector = einops.einsum(to_vector, to_vector, "batch seq d_model, batch seq d_model -> batch seq")
    
    projected_lengths = (dot_product) / (length_of_vector)
    projections = to_vector * einops.repeat(projected_lengths, "batch seq -> batch seq d_model", d_model = to_vector.shape[-1])
    return projections


# %% Code to first intervene by subtracting output in residual stream
def add_vector_to_resid(
    original_resid_stream: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "batch d_model"],
    positions = Union[Float[Tensor, "batch"], int],
) -> Float[Tensor, "batch n_head pos pos"]:
    '''
    Hook that adds vector into residual stream at position
    '''
    assert len(original_resid_stream.shape) == 3
    assert len(vector.shape) == 2
    assert original_resid_stream.shape[0] == vector.shape[0]
    assert original_resid_stream.shape[2] == vector.shape[1]
    
    if isinstance(positions, int):
        device = "cuda" if original_resid_stream.get_device() >= 0 else "cpu"
        positions = torch.tensor([positions] * original_resid_stream.shape[0]).to(device)
    
    
    expanded_positions = einops.repeat(positions, "batch -> batch 1 d_model", d_model = vector.shape[1])
    resid_stream_at_pos = torch.gather(original_resid_stream, 1, expanded_positions)
    resid_stream_at_pos = einops.rearrange(resid_stream_at_pos, "batch 1 d_model -> batch d_model")
    
    resid_stream_at_pos = resid_stream_at_pos + vector
    for i in range(original_resid_stream.shape[0]):
        original_resid_stream[i, positions[i], :] = resid_stream_at_pos[i]
    return original_resid_stream
# %%
def add_vector_to_all_resid(
    original_resid_stream: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    vector: Float[Tensor, "batch pos d_model"],
) -> Float[Tensor, "batch n_head pos pos"]:
    '''
    Hook that just adds a vector to the entire residual stream
    '''
    assert len(original_resid_stream.shape) == 3
    assert original_resid_stream.shape == vector.shape
    
    original_resid_stream = original_resid_stream + vector
    return original_resid_stream
   
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False 
    
    
def replace_output_hook(
    original_output: Float[Tensor, "batch seq head d_model"],
    hook: HookPoint,
    new_output: Float[Tensor, "batch seq d_model"],
    head: int,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook that replaces the output of a head with a new output
    '''
    
    assert len(original_output.shape) == 4
    assert len(new_output.shape) == 3
    assert original_output.shape[0] == new_output.shape[0]
    assert original_output.shape[1] == new_output.shape[1]
    assert original_output.shape[3] == new_output.shape[2]
    
    original_output[:, :, head, :] = new_output
    
    return original_output

def replace_output_of_specific_batch_pos_hook(
    original_output: Float[Tensor, "batch seq head d_model"],
    hook: HookPoint,
    new_output: Float[Tensor, "d_model"],
    head: int,
    batch: int,
    pos: int,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook that replaces the output of a head on a batch/pos with a new output
    '''
    #print(original_output.shape)
    #print(new_output.shape)
    assert len(original_output.shape) == 4
    assert len(new_output.shape) == 1
    assert original_output.shape[-1] == new_output.shape[-1]
    
    original_output[batch, pos, head, :] = new_output
    
    return original_output

def replace_output_of_specific_MLP_batch_pos_hook(
    original_output: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    new_output: Float[Tensor, "d_model"],
    batch: int,
    pos: int,
) -> Float[Tensor, "batch seq d_model"]:
    '''
    Hook that replaces the output of a MLP layer on a batch/pos with a new output
    '''
    
    assert len(original_output.shape) == 3
    assert len(new_output.shape) == 1
    assert original_output.shape[-1] == new_output.shape[-1]
    
    original_output[batch, pos, :] = new_output
    return original_output


def get_item_hook(
    item,
    hook: HookPoint,
    storage: list
):
    '''
    Hook that just returns this specific item.
    '''
    storage.append(item)
    return item


def replace_model_component_completely(
    model_comp,
    hook: HookPoint,
    new_model_comp,
):
    if isinstance(model_comp, torch.Tensor):
        model_comp[:] = new_model_comp
    return new_model_comp



def get_single_correct_logit(logits: Float[Tensor, "batch pos d_vocab"],
                     batch: int,
                     pos: int,
                     tokens: Float[Tensor, "batch pos"]):
    """
    get the correct logit at a specific batch and position (of the next token)
    """
    
    correct_next_token = tokens[batch, pos + 1]
    return logits[batch, pos, correct_next_token]
    
    
# Function to generate a dataset function 
def dataset_generator(dataset_tokens, batch_size, prompt_len):
    total_prompts = dataset_tokens.shape[0]
    num_batches = total_prompts // batch_size

    #print(num_batches)
    for batch_idx in range(num_batches):
        clean_batch_offset = batch_idx * batch_size
        start_clean_prompt = clean_batch_offset
        end_clean_prompt = clean_batch_offset + batch_size
        
        corrupted_batch_offset = (batch_idx + 1) * batch_size
        start_corrupted_prompt = corrupted_batch_offset
        end_corrupted_prompt = corrupted_batch_offset + batch_size

        clean_tokens = dataset_tokens[start_clean_prompt:end_clean_prompt, :prompt_len]
        corrupted_tokens = dataset_tokens[start_corrupted_prompt:end_corrupted_prompt, :prompt_len]
        #print(corrupted_tokens.shape[0])
        if corrupted_tokens.shape[0] == 0:
            corrupted_tokens = dataset_tokens[:batch_size, :prompt_len] # loop it back around
        
        
        yield batch_idx, clean_tokens, corrupted_tokens


def prepare_dataset(model, device, TOTAL_TOKENS: int, BATCH_SIZE, PROMPT_LEN, padding: bool, dataset_name = "pile"):
    """
    returns the dataset, and the number of prompts in this dataset
    """
    dataset = utils.get_dataset(dataset_name)
    
    if not padding:
        new_dataset = utils.tokenize_and_concatenate(dataset, model.tokenizer, max_length=PROMPT_LEN)
        all_dataset_tokens = new_dataset['tokens'].to(device)
    else:
        print("Not complete yet")
        all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
        PAD_TOKEN = model.to_tokens(model.tokenizer.pad_token)[-1, -1].item() 
        

    assert len(all_dataset_tokens.shape) == 2
    total_prompts = TOTAL_TOKENS // (PROMPT_LEN)
    num_batches = total_prompts // BATCH_SIZE
    
    if(num_batches <= 1):
        raise ValueError("Need to have more than 2 batches for corrupt prompt gen to work")
    
    # Create the generator
    dataset = dataset_generator(all_dataset_tokens[:total_prompts], BATCH_SIZE, PROMPT_LEN)
    
    #print(num_batches)
    return dataset, num_batches

