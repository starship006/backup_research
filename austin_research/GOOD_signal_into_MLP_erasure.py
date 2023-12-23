"""
Across most models, the MLP layers seem to be performing some sort of erasure. I want to understand what signal is feeding into this, and have more data on what is
going on.
"""
# %%
import sys

from imports import *
from updated_nmh_dataset_gen import generate_ioi_prompts
from GOOD_helpers import collect_direct_effect
from reused_hooks import overwrite_activation_hook, add_and_replace_hook, store_item_in_tensor

%load_ext autoreload
%autoreload 2
from GOOD_helpers import *
 

# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    print("Running in notebook mode")
    #%load_ext autoreload
    #%autoreload 2
    model_name = "pythia-410m"
else:
    print("Running in script mode")
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name
    
# %%
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
batch_size = 50 # 30 -- if model large
prompt_len = 30 # 15 -- if model large

owt_dataset = utils.get_dataset("owt")
owt_dataset_name = "owt"    
    
all_owt_tokens = model.to_tokens(owt_dataset[0:(batch_size * 2)]["text"]).to(device)
owt_tokens = all_owt_tokens[0:batch_size][:, :prompt_len]
corrupted_owt_tokens = all_owt_tokens[batch_size:batch_size * 2][:, :prompt_len]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (batch_size, prompt_len)

# %%
clean_tokens: Float[Tensor, "batch pos"] = owt_tokens
# %%
logits, cache = model.run_with_cache(clean_tokens)
# %% First, see which heads in the model are even useful for predicting these (across ALL positions_)
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)

if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")

correct_logit = get_correct_logit_score(logits, owt_tokens)
# %%
def make_scatter(name_data_tuple, x_data, x_label = "Direct Effect of Head", 
                 plot_y_equal_x = False, plot_y_equal_minus_x = True, indices = None, title = None):
    traces = []
    if indices is not None:
        indexed_x_data = x_data[indices]
    else:
        indexed_x_data = x_data
    
    hover_texts = [f'Index: {i}' for i in range(len(indexed_x_data))]
    
    for y_data, name in name_data_tuple:
        assert type(name) == str
        if indices is not None:
            indexed_y_data = y_data[indices]
        else:
            indexed_y_data = y_data
        
        #print(f"len(x_data) = {len(indexed_x_data)}")
        #print(f"len(y_data) = {len(indexed_y_data)}")
        assert len(indexed_x_data) == len(indexed_y_data)
        
        traces.append(go.Scatter(
            x = indexed_x_data,
            y = indexed_y_data,
            mode = 'markers',
            name = name,
            marker = dict(size = 2),
            text = hover_texts,  # Add the hover text to the trace
            hoverinfo = 'text+x+y'  # Display hover text along with x and y values  
        ))
    
    x_line = [min(indexed_x_data), max(indexed_x_data)]
    data = traces
    
    if plot_y_equal_minus_x:
        y_line = [-x for x in x_line]
        trace3 = go.Scatter(
            x = x_line,
            y = y_line,
            mode = 'lines',
            name = 'y = -x'
        )
    
        data = data + [trace3]
        
    if plot_y_equal_x:
        y_line = x_line
        trace3 = go.Scatter(
            x = x_line,
            y = y_line,
            mode = 'lines',
            name = 'y = x'
        )
        data = data + [trace3]
    
    if title is None:
        title = f'Effects of Ablating Head {head_to_ablate}'

    
    layout = go.Layout(
        title = title,
        xaxis = dict(title = x_label),
        yaxis = dict(title = 'Measured Values'),
        hovermode = 'closest'
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

def project_away_directions_of_almost_all_positions(
    original_heads_out: Float[Tensor, "batch seq head_idx d_model"],
    hook: HookPoint,
    project_away_vector: Float[Tensor, "batch seq_minus_one d_model"],
    heads = [], # array of ints,
    project_only = False # whether to, instead of projecting away the vector, keep it!
) -> Float[Tensor, "batch n_head pos pos"]:
    '''
    Function which gets removes a specific component (or keeps only it, if project_only = True) of the an output of a head
    for all positions except the last 
    '''
    # right now this projects away the IO direction!
    assert len(original_heads_out.shape) == 4 and len(project_away_vector.shape) == 3 
    assert original_heads_out.shape[0] == project_away_vector.shape[0] 
    assert original_heads_out.shape[1] == project_away_vector.shape[1] + 1
    assert original_heads_out.shape[-1] == project_away_vector.shape[-1]
    
    for head in heads:
        head_output: Float[Tensor, "batch seq_minus_one d_model"] = original_heads_out[:, :-1, head, :]
        projections = get_3d_projection(head_output, project_away_vector)

        if project_only:
            resid_without_projection = projections
        else:
            resid_without_projection = (head_output - projections) 
        
        original_heads_out[:, :-1, head, :] = resid_without_projection

    return original_heads_out
# %%
head_to_ablate = (22, 11)
FREEZE_FINAL_LN = False

# %% First, get normal ablation self-repair
model.reset_hooks()
new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = True)
model.reset_hooks()
new_logits = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = False)

# %%  Get self-repair from projection away correct token
token_dirs = model.tokens_to_residual_directions(owt_tokens[:, 1:])
KEEP_TOKEN_DIRECTION = True

model.reset_hooks()
partial_hook = partial(project_away_directions_of_almost_all_positions, heads = [head_to_ablate[1]], project_away_vector = token_dirs, project_only = KEEP_TOKEN_DIRECTION)
model.add_hook(utils.get_act_name("result", head_to_ablate[0]), partial_hook)
projected_logits, projected_cache = model.run_with_cache(owt_tokens)
model.reset_hooks()


# %%
if FREEZE_FINAL_LN:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
    projected_per_head_direct_effect, projected_all_layer_direct_effect = collect_direct_effect(projected_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)    
else:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False)
    projected_per_head_direct_effect, projected_all_layer_direct_effect = collect_direct_effect(projected_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False)



ablated_all_head_diff: Float[Tensor, "layer head batch pos"] = (ablated_per_head_direct_effect - per_head_direct_effect)
ablated_all_mlp_diff: Float[Tensor, "layer batch pos"] = (ablated_all_layer_direct_effect - all_layer_direct_effect)

projected_all_head_diff: Float[Tensor, "layer head batch pos"] = (projected_per_head_direct_effect - per_head_direct_effect)
projected_all_mlp_diff: Float[Tensor, "layer batch pos"] = (projected_all_layer_direct_effect - all_layer_direct_effect)


ablated_correct_logit = get_correct_logit_score(new_logits, owt_tokens)
projected_correct_logit = get_correct_logit_score(projected_logits, owt_tokens) #type:ignore

ablated_diff_in_logits = (ablated_correct_logit - correct_logit)
projected_diff_in_logits = (projected_correct_logit - correct_logit)

# %%
ablated_change_in_de_all_layers = ablated_all_mlp_diff.sum((0))
projected_change_in_de_all_layers = projected_all_mlp_diff.sum((0))

# %% Compare the pure self-repair first
x_data = per_head_direct_effect[head_to_ablate].flatten().cpu()

make_scatter([(ablated_diff_in_logits.flatten().cpu(), "Change in Correct Logit Score - Ablation"),
             (projected_diff_in_logits.flatten().cpu(), "Change in Correct Logit Score - Projection"),],
             x_data, title = f"Change in Logits when Ablating vs Projecting Head {head_to_ablate} | Keep Tokens == {KEEP_TOKEN_DIRECTION}")


# %% Look at self-repair in MLP layers
make_scatter([(ablated_change_in_de_all_layers.flatten().cpu(), "Change in DE of all layers - Ablation"),
              (projected_change_in_de_all_layers.flatten().cpu(), "Change in DE of all layers - Projection"),]
             , x_data, plot_y_equal_x = True)
# %%
