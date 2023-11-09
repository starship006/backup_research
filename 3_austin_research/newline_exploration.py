# %%
"""
The idea behind this notebook is to investigate a phenomena. I observed in pythia160 million, and potentially other models,
where uploading a prediction of a new line can lead to weird model compensations. I had observed not just soft repair, but increased mama performance whenever you be late model on new lines.
I don't quite know why this may be happening. One thing I recall, though, is that it occurred whenever they were two newlines in a row.
My base hypothesis, for this is that the model believes that after one newline, there will not be another newline, so it writes against it.

Ablating this just seems to be ablating the suppression of a newline. I can quickly test this and see whats up.

Wait. This is immediaely wrong.
"""
# %%
from imports import *
import argparse
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
from beartype import beartype as typechecker
from typing import List, Tuple
import torch
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
owt_dataset_name = "owt"

# Constants


LENGTH_PROMPT_BEFORE_NEWLINE = 15 # Length of the everything before first newline (including the added period)
LENGTH_PROMPT_AFTER_NEWLINE = 15
PROMPT_LEN = LENGTH_PROMPT_BEFORE_NEWLINE + LENGTH_PROMPT_AFTER_NEWLINE + 2  # +2 for the two newlines
BATCH_SIZE = 130
ALTERNATIVE_TOKEN = model.to_tokens("$", prepend_bos = False).item()  # This is the token we will use to replace accidental NEWLINE_TOKENs
all_owt_tokens = model.to_tokens(owt_dataset[0:(1000)]["text"])

STYLE_OPTIONS = ["double_newline", "double_newline_period", "double_period", "owt"]
STYLE = STYLE_OPTIONS[0]


def generate_tokens(all_owt_tokens, STYLE, BATCH_SIZE, LENGTH_PROMPT_BEFORE_NEWLINE, ALTERNATIVE_TOKEN):
    NEWLINE_TOKEN = model.to_tokens("\n", prepend_bos = False).item()
    PERIOD_TOKEN = model.to_tokens(".", prepend_bos = False).item()
    MAX_TOKEN = model.cfg.d_vocab
    
    replaced_token = None
    if STYLE == "double_newline" or STYLE == "double_newline_period":
        replaced_token = NEWLINE_TOKEN
    elif STYLE == "double_period":
        replaced_token = PERIOD_TOKEN
    elif STYLE == "owt":
        all_owt_tokens = model.to_tokens(owt_dataset[0:(BATCH_SIZE * 2)]["text"]).to(device)
        clean_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
        corrupted_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
        return clean_tokens, corrupted_tokens
    else:
        raise Exception("STYLE not recognized")


    # Helper function to replace accidental NEWLINE_TOKENs
    def replace_tokens(tensor, replacement_token, replaced_token = NEWLINE_TOKEN):
        """
        replaces replaced_token with replacement_token
        """
        mask = tensor == replaced_token
        tensor[mask] = replacement_token
        return tensor


    # Generate reasonably random tokens before the period and replace accidental NEWLINE_TOKENs
    random_indicies = torch.randperm(all_owt_tokens.shape[0])[:BATCH_SIZE]
    tokens_before_newline = all_owt_tokens[random_indicies, 0:LENGTH_PROMPT_BEFORE_NEWLINE]
    tokens_before_newline = replace_tokens(tokens_before_newline, ALTERNATIVE_TOKEN)

    # Add period to end if of subsection if needed
    if STYLE == "double_newline_period":
        tokens_before_newline[:, -1] = PERIOD_TOKEN

    # Generate random tokens after the newline and replace accidental NEWLINE_TOKENs (but start from 1 to avoid endoftext)
    random_indicies = torch.randperm(all_owt_tokens.shape[0])[:BATCH_SIZE]
    tokens_after_newline = all_owt_tokens[random_indicies, 1:LENGTH_PROMPT_BEFORE_NEWLINE + 1]

    # Replace accidental tokens with another
    tokens_after_newline = replace_tokens(tokens_after_newline, ALTERNATIVE_TOKEN, replaced_token=replaced_token)

    # Add period to end if of subsection if needed
    if STYLE == "double_newline_period":
        tokens_after_newline[:, -1] = PERIOD_TOKEN

    # Concatenate the two parts with middle tokens
    mid_token = None
    if STYLE == "double_newline" or STYLE == "double_newline_period":
        mid_token = torch.full((BATCH_SIZE, 1), NEWLINE_TOKEN,  device = device)
    elif STYLE == "double_period":
        mid_token = torch.full((BATCH_SIZE, 1), PERIOD_TOKEN,  device = device)
    
    clean_tokens = torch.cat((tokens_before_newline, mid_token, mid_token, tokens_after_newline), dim=1)

    # Generate corrupted tokens by replacing the NEWLINE_TOKEN with a random token
    corrupted_tokens = clean_tokens.clone()
    corrupted_tokens[corrupted_tokens == NEWLINE_TOKEN] = torch.randint(1, MAX_TOKEN, (1,)).item()
    corrupted_tokens = replace_tokens(corrupted_tokens, ALTERNATIVE_TOKEN, replaced_token=replaced_token)

    clean_tokens = clean_tokens.to(device)
    corrupted_tokens = corrupted_tokens.to(device)


    assert clean_tokens.shape == corrupted_tokens.shape
    assert clean_tokens.shape[0] == BATCH_SIZE
    assert clean_tokens.shape[1] == PROMPT_LEN
    return clean_tokens, corrupted_tokens

# %%
clean_tokens, corrupted_tokens = generate_tokens(all_owt_tokens, STYLE, BATCH_SIZE, LENGTH_PROMPT_BEFORE_NEWLINE, ALTERNATIVE_TOKEN)


# %%
logits, cache = model.run_with_cache(clean_tokens)

print(utils.lm_accuracy(logits, clean_tokens))
print(utils.lm_cross_entropy_loss(logits, clean_tokens))
# %% Import helper functions
partials = return_partial_functions(model = model, clean_tokens = clean_tokens, corrupted_tokens = corrupted_tokens, cache = cache)
globals().update(partials)
# %%
per_head_direct_effect, per_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, display = in_notebook_mode, collect_individual_neurons = False)
# %%
show_input(per_head_direct_effect.mean((-1,-2)), per_layer_direct_effect.mean((-1,-2)))
# %%
@jaxtyped
@typechecker
def print_tokens(batch: int, start: int = 40, end: int = 47):
    """
    Prints the tokens for a batch. Shares same indexing.
    """
    print("Tokens before:")
    print(model.to_string(clean_tokens[batch, 0:start]))
    print("Start token == " , model.to_string(clean_tokens[batch, start]))
    print("Tokens after:")
    print(model.to_string(clean_tokens[batch, start + 1:end]))
    # print("...")
    # print(model.to_string(all_owt_tokens[batch, end:]))


@jaxtyped
@typechecker
def zero_ablation_hook(
    attn_result: Union[Float[Tensor, "batch seq n_heads d_model"], Float[Tensor, "batch seq d_model"]],
    hook: HookPoint,
    head_index_to_ablate: int = -1000,
) -> Union[Float[Tensor, "batch seq n_heads d_model"], Float[Tensor, "batch seq d_model"]]:
    """
    zero ablates a head or mlp layer across all batch x positions
    """

    if len(attn_result.shape) == 3:
        attn_result[:] = torch.zeros(attn_result.shape)
    else:
        # attention head
        attn_result[:, :, head_index_to_ablate, :] = torch.zeros(attn_result[:, :, head_index_to_ablate, :].shape)
    return attn_result

def replace_ln(
    current_ln_scaling: Float[Tensor, "batch seq 1"],
    hook: HookPoint,
    hook_ln_scaling: Float[Tensor, "batch seq 1"]
):
    """
    replaces the scaling of the layer norm with new_ln_scaling
    """
    current_ln_scaling[:] = hook_ln_scaling
    return hook_ln_scaling

@jaxtyped
@typechecker
def one_component_zero_ablate(
                    clean_tokens,
                    clean_cache: ActivationCache,
                    ablate_final_ln: bool = True,
                    attention_head: Tuple[int, int] = None,
                    mlp_layer: int = None,
                    ) -> Float[Tensor, "batch pos d_vocab"] : 
    """
    temp ablation function to test if the thing with activation patching is just layernorm being frozen or not
    pass in an attention head in the last layer
    """
    model.reset_hooks()
    

    if attention_head != None:
        model.add_hook(utils.get_act_name("z", attention_head[0]), partial(zero_ablation_hook, head_index_to_ablate = attention_head[1]))
    if mlp_layer != None:
        model.add_hook(utils.get_act_name("mlp_out", mlp_layer),zero_ablation_hook)
    model.add_hook(f'ln_final.hook_scale', partial(replace_ln, hook_ln_scaling = clean_cache[f'ln_final.hook_scale']))

    logits, _ = model.run_with_cache(clean_tokens)

    return logits

@jaxtyped
@typechecker
def sample_ablate_new_logits_calc(heads: Union[List[Tuple[int, int]], None] = None,
                                  mlp_layers: Union[List[int], None] = None,
                                  num_runs: int = 5, logits: Float[Tensor, "batch pos d_vocab"] = logits,
                                  freeze_final_ln = False) -> torch.Tensor:
    """
    runs activation patching over component and returns new avg_correct_logit_score, averaged over num_runs runs
    this is the average logit of the correct token upon num_runs sample ablations
    logits just used for size
    """
    assert num_runs > 0
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    # use many shuffled!
    logits_accumulator = torch.zeros_like(logits)
    for i in range(num_runs):
        # Shuffle clean_tokens by batch
        shuffled_corrupted_tokens = shuffle_owt_tokens_by_batch(corrupted_tokens)
        # Calculate new_logits using act_patch
        if freeze_final_ln:
            pass
        else:
            new_logits = act_patch(model, clean_tokens, nodes, return_item, shuffled_corrupted_tokens, apply_metric_to_cache=False)
        logits_accumulator += new_logits

    avg_logits = logits_accumulator / num_runs
    # get change in direct effect compared to original
    #avg_correct_logit_score = get_correct_logit_score(avg_logits)
    return avg_logits



@jaxtyped
@typechecker
def ablation_new_logits_wrapper(heads: Union[List[Tuple[int, int]], None] = None,
                                  mlp_layers: Union[List[int], None] = None,
                                  num_runs: int = 5, logits: Float[Tensor, "batch pos d_vocab"] = logits,
                                  ablation_type: str = "sample",
                                  freeze_final_ln = False) -> torch.Tensor:
    if num_runs != 5:
        print("WARNING: num_runs != 5")
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        #print(layer)
    if mlp_layers is not None and heads is None:
        assert len(set(mlp_layers)) == 1
    if mlp_layers is not None and heads is not None:
        raise Exception("Can't have both heads and mlp_layers")
    if heads is None and mlp_layers is None:
        raise Exception("No heads or mlp layers given")
    

    # key = "H:" + str(heads) +  "M:" + str(mlp_layers)
    # if key in new_logits_after_sample:
    #     print("Returning cached activation")
    #     return new_logits_after_sample[key]
    
    if ablation_type == "sample":
        result = sample_ablate_new_logits_calc(heads, mlp_layers, num_runs, logits)
    elif ablation_type == "zero":
        nodes = []
        if heads != None :
            nodes += [Node("z", layer, head) for (layer,head) in heads]
        if mlp_layers != None:
            nodes += [Node("mlp_out", layer) for layer in mlp_layers]

        if freeze_final_ln:
            if (heads != None and len(heads) > 1) or (mlp_layers != None and len(mlp_layers) > 1):
                raise Exception("Can't freeze final ln with more than one head/mlp (change the code if you want to)")
            if heads != None:
                result = one_component_zero_ablate(clean_tokens, cache, attention_head=heads[0])
            elif mlp_layers != None:
                result = one_component_zero_ablate(clean_tokens, cache, mlp_layer = mlp_layers[0])
        else:
            result = act_patch(model, clean_tokens, nodes, return_item, new_cache="zero", apply_metric_to_cache= False)        
    else:
        raise Exception("ablation_type not recognized")
    
    # new_logits_after_sample[key] = result
    
    # with open(SAMPLE_ABLATE_CORRECT_LOGIT_DIR_NAME, 'wb') as f:
    #     pickle.dump(new_logits_after_sample, f)
        
    return result

def show_batch_result(batch, start = 40, end = 47, per_head_direct_effect = per_head_direct_effect, per_layer_direct_effect = per_layer_direct_effect):
    """
    highlights the text selection, along with the mean effect of the range
    indexed similariy to python, where start is inclusive and end is exclusive 

    recall that the per_head_direct_effect is one length shorter than the input, since it doesn't have the first token
    so, if the interesting self-repair you are observing seems to be at pos 12, this means it is for the prediction of token 13
    """
    print("unchecked")
    print_tokens(start, end)
    show_input(per_head_direct_effect[..., batch, start:end].mean(-1),per_layer_direct_effect[:, batch, start:end].mean(-1), title = f"Direct Effect of Heads on batch {batch}")

def shuffle_owt_tokens_by_batch(clean_tokens):
    """
    given a tensor of shape (batch_size, num_tokens), shuffles the batches
    """
    batch_size, num_tokens = clean_tokens.shape

    assert batch_size == BATCH_SIZE
    assert num_tokens == PROMPT_LEN

    shuffled_owt_tokens = torch.zeros_like(clean_tokens)
    
    
    perm = torch.randperm(batch_size)
    shuffled = clean_tokens[perm].clone()
    return shuffled

def create_scatter_of_change_from_component(heads = None, mlp_layers = None, return_slope = False, zero_ablate = False, force_through_origin = False, num_runs = 1, logits = logits, focus_on_newline_to_newline = False, 
                                            freeze_final_ln = False):
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
     

    
    new_logits = ablation_new_logits_wrapper(heads, mlp_layers, num_runs, logits, 
                                             ablation_type= "zero" if zero_ablate else "sample",
                                             freeze_final_ln = freeze_final_ln)

    
    assert new_logits.shape == logits.shape
    assert len(new_logits.shape) == 3  # should be [batch x pos x vocab_size]
    diff_in_logits = new_logits - logits
    change_in_direct_effect = get_correct_logit_score(diff_in_logits, clean_tokens=clean_tokens)

    
    #  3) plots the clean direct effect vs accumulated backup
    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else per_layer_direct_effect[layer].flatten().cpu()
    change_in_direct_effect = change_in_direct_effect.flatten().cpu()
    assert direct_effects.shape == change_in_direct_effect.shape
    assert direct_effects.shape[0] == BATCH_SIZE * (PROMPT_LEN - 1)
    


    # If focus_on_newline_to_newline, only consider the direct effect of the newline to newline
    if focus_on_newline_to_newline:
        indicies = [i for i in range(BATCH_SIZE * (PROMPT_LEN - 1)) if i % (PROMPT_LEN - 1) == LENGTH_PROMPT_BEFORE_NEWLINE]
        direct_effects = direct_effects[indicies]
        change_in_direct_effect = change_in_direct_effect[indicies]


    # get a best fit line
    if force_through_origin:
        slope = np.linalg.lstsq(direct_effects.reshape(-1, 1), change_in_direct_effect, rcond=None)[0][0]
        intercept = 0
    else:
        slope, intercept = np.linalg.lstsq(np.vstack([direct_effects, np.ones(len(direct_effects))]).T, change_in_direct_effect, rcond=None)[0]
    
    if not return_slope:
        fig = go.Figure()
        text_labels = [f"Batch {i[0]}, Pos {i[1]}: {model.to_string(clean_tokens[i[0], i[1]:(i[1] + 1)])} --> {model.to_string(clean_tokens[i[0], (i[1] + 1):(i[1] + 2)])}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]

        # change text_labels to only show newline to newline
        if focus_on_newline_to_newline:
            indicies = [i for i in range(BATCH_SIZE * (PROMPT_LEN - 1)) if i % (PROMPT_LEN - 1) == LENGTH_PROMPT_BEFORE_NEWLINE]
            text_labels = [text_labels[i] for i in indicies]

        scatter_plot = go.Scatter(
            x = direct_effects,
            y = change_in_direct_effect,
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
        fig.update_layout(width=1100, height=500)
        fig.show()
    
    if return_slope:
        return slope.item()
    


def get_threshold_from_percent(logits, threshold_percent_filter):
    logits_flat = logits.flatten()
    sorted_logits = logits_flat.sort()[0]
    index = int((1 - threshold_percent_filter) * sorted_logits.size(0))
    threshold_value = sorted_logits[index]
    return threshold_value
def get_top_self_repair_prompts(heads = None, mlp_layers = None, topk = 10, num_runs = 5, logits = logits, logit_diff_self_repair = True, threshold_percent_filter = 1):
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
        direct_effect = per_layer_direct_effect[layer]
    else:
        raise Exception("No heads or mlp layers given")
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    # get change in direct effect compared to original
    print("CHECK THIS FUNCTION FIRST; THERE MAY BE WEIRD STUFF GOING ON WITH BROADCASTING")
    change_in_direct_effect = ablation_new_logits_wrapper(heads, mlp_layers, num_runs, logits) - get_correct_logit_score(logits)
    
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

def get_cde_each_head(target_pos = None, ablate_type = "sample", freeze_final_ln = False):
    """
    returns a tensor of shape (n_layers, n_heads) with the change in direct effect for each head
    target_pos: if not none, will only return the change in direct effect for that position
    """

    cde_each_head = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            new_logits = ablation_new_logits_wrapper(heads = [(layer, head)], num_runs = 5, logits = logits, ablation_type = ablate_type, freeze_final_ln = freeze_final_ln)
            diff_in_logits = new_logits - logits
            change_in_direct_effect: Float[Tensor, "batch, pos - 1"] = get_correct_logit_score(diff_in_logits)
            if target_pos is not None:
                cde_each_head[layer, head] = change_in_direct_effect[:, target_pos].mean(-1)
            else:
                cde_each_head[layer, head] = change_in_direct_effect.mean((-1,-2))
    return cde_each_head

def get_cde_each_mlp_layer(target_pos = None, ablate_type = "sample", freeze_final_ln = False):
    """
    returns a tensor of shape (n_layers,) with the change in direct effect for each layer
    target_pos: if not none, will only return the change in direct effect for that position
    """
    cde_each_layer = torch.zeros((model.cfg.n_layers))
    for layer in range(model.cfg.n_layers):
        new_logits = ablation_new_logits_wrapper(mlp_layers = [layer], num_runs = 5, logits = logits, ablation_type = ablate_type, freeze_final_ln = freeze_final_ln)
        diff_in_logits = new_logits - logits
        change_in_direct_effect: Float[Tensor, "batch, pos - 1"] = get_correct_logit_score(diff_in_logits)
        if target_pos is not None:
            cde_each_layer[layer] = change_in_direct_effect[:, target_pos].mean(-1).mean()
        else:
            cde_each_layer[layer] = change_in_direct_effect.mean((-1,-2)).mean()
    return cde_each_layer




@jaxtyped
@typechecker
def create_layered_scatter(
    heads_x: Float[Tensor, "layer head"],
    heads_y: Float[Tensor, "layer head"], 
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

    # Add colorbar if MLP data is included
    # if mlp_x is not None and mlp_y is not None:
    #     fig.update_layout(
    #         coloraxis_colorbar=dict(
    #             title='Layer',
    #             tickvals=[0, 1],
    #             ticktext=['Head', 'MLP'],
    #             lenmode='fraction', 
    #             len=0.75,
    #             yanchor='middle',
    #             y=0.5,
    #             # Adjust the position of the colorbar to prevent overlap
    #             xanchor='left',  # Anchor the colorbar to the left
    #             x=3  # Slightly increase the x position to move it away from the main plot
    #         )
    #     )

    fig.show()



# %%
create_scatter_of_change_from_component(heads = [(11,1)], force_through_origin=True, num_runs = 5, zero_ablate=True, focus_on_newline_to_newline=False, freeze_final_ln=True)
# %%
create_scatter_of_change_from_component(mlp_layers=[11], force_through_origin=True, num_runs = 5, zero_ablate=True, focus_on_newline_to_newline=False, freeze_final_ln=True)
# %% See what are the average direct effects of the newline to newline
show_input(per_head_direct_effect[..., LENGTH_PROMPT_BEFORE_NEWLINE].mean(-1), 
           per_layer_direct_effect[..., LENGTH_PROMPT_BEFORE_NEWLINE].mean(-1),
           title = f"Average Target DE for style == {STYLE}")
# %%

# %%

slopes = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        slopes[layer, head] = create_scatter_of_change_from_component(heads = [(layer, head)], force_through_origin=True, num_runs = 5, zero_ablate=True, return_slope=True, focus_on_newline_to_newline=True, freeze_final_ln=True)
# %%
imshow(slopes, title = f"Slopes for style == {STYLE}")
# %%
ablate_type = "zero"
per_head_cde: Float[Tensor, "layer head"] = get_cde_each_head(ablate_type = ablate_type, freeze_final_ln = True)
per_layer_cde: Float[Tensor, "layer"] = get_cde_each_mlp_layer(ablate_type = ablate_type, freeze_final_ln = True)
# %% Create a scatter of per_head_direct_effect vs per_head_cde
create_layered_scatter( 
    heads_x = per_head_direct_effect.mean((-1, -2)), 
    heads_y = per_head_cde, 
    x_title = 'Direct Effect of Head', 
    y_title = 'Change in Direct Effect', 
    plot_title = f'Direct Effect vs Change in Direct Effect across all positions in {safe_model_name}; zero_ablate = {ablate_type == "zero"}',
    #mlp_x = per_layer_direct_effect.mean((-1, -2)),
    #mlp_y = per_layer_cde
) 

# %%
ablate_type = "zero"
per_head_cde_newline: Float[Tensor, "layer head"] = get_cde_each_head(target_pos = LENGTH_PROMPT_BEFORE_NEWLINE, ablate_type = ablate_type)
per_layer_cde_newline: Float[Tensor, "layer"] = get_cde_each_mlp_layer(target_pos = LENGTH_PROMPT_BEFORE_NEWLINE, ablate_type = ablate_type)

# %%
create_layered_scatter(
    heads_x=per_head_direct_effect[..., LENGTH_PROMPT_BEFORE_NEWLINE].mean((-1)), 
    heads_y=per_head_cde_newline, 
    x_title='Direct Effect of Head', 
    y_title='Change in Direct Effect',
    plot_title=f'Direct Effect vs Change in Direct Effect for \\n -> \\n in {safe_model_name}; zero_ablate = {ablate_type == "zero"}',
    #mlp_x=per_layer_direct_effect[..., LENGTH_PROMPT_BEFORE_NEWLINE].mean((-1)),
    #mlp_y=per_layer_cde_newline
)
# %% There is overlap in the ones that don't get backed up / the ones that don't get summarized

smallest_one_eight = topk_of_Nd_tensor(-1 * per_head_direct_effect[1,8,:, LENGTH_PROMPT_BEFORE_NEWLINE], 15)
smallest_mlp_eleven = topk_of_Nd_tensor(-1 * per_layer_direct_effect[11,:, LENGTH_PROMPT_BEFORE_NEWLINE], 15)
# %%
just_indicies_one_eight = set([i[0] for i in smallest_one_eight])
just_indicies_mlp_eleven = set([i[0] for i in smallest_mlp_eleven])

# %%
shared_elements = just_indicies_one_eight.intersection(just_indicies_mlp_eleven)
number_of_shared_elements = len(shared_elements)
print(number_of_shared_elements)
print(len(just_indicies_one_eight))

# %%
