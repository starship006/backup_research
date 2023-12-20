"""
This code does preliminary exploration of what components are responsile for self-repairing logits.
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
model.set_use_attn_result(False)
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
head_to_ablate = (22, 11)
model.reset_hooks()
new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = True)
model.reset_hooks()
new_logits = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = False)
ablated_correct_logit = get_correct_logit_score(new_logits, owt_tokens)

FREEZE_FINAL_LN = False
if FREEZE_FINAL_LN:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
else:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False)



all_head_diff: Float[Tensor, "layer head batch pos"] = (ablated_per_head_direct_effect - per_head_direct_effect)
all_mlp_diff: Float[Tensor, "layer batch pos"] = (ablated_all_layer_direct_effect - all_layer_direct_effect)

# %%
diff_in_logits = (ablated_correct_logit - correct_logit)

#histogram((diff_in_logits).flatten(), title = "Change in Correct Logit Score")
#scatter(per_head_direct_effect[head_to_ablate].flatten(), diff_in_logits.flatten(), title = "DE of head vs Change in Correct Logit Score")

self_repair_displayed = diff_in_logits + per_head_direct_effect[head_to_ablate]
change_in_de_all_heads = all_head_diff.sum((0, 1))
change_in_de_all_layers = all_mlp_diff.sum((0))


#scatter(per_head_direct_effect[head_to_ablate].flatten(), self_repair_displayed.flatten(), title = "DE of head vs Self Repair (measured in logits)")

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
        title = f'Effects of Ablating Head {head_to_ablate} | ln frozen for DE = {FREEZE_FINAL_LN}'

    
    layout = go.Layout(
        title = title,
        xaxis = dict(title = x_label),
        yaxis = dict(title = 'Measured Values'),
        hovermode = 'closest'
    )
    fig = go.Figure(data=data, layout=layout)
    fig.show()

# %%
per_head_direct_effect_flatten = per_head_direct_effect[head_to_ablate].flatten().cpu()
change_in_de_total = (change_in_de_all_heads + change_in_de_all_layers).flatten().cpu()

x_data = per_head_direct_effect_flatten
y_data_1 = diff_in_logits.flatten().cpu()
y_data_2 = change_in_de_total

name_one = "DE of head vs Change in Logits"
name_two = 'DE of head vs Change in DE of internal components'

make_scatter([(y_data_1, name_one), (y_data_2, name_two)], x_data)
# %%
change_in_all_heads_except_ablated = change_in_de_all_heads.flatten().cpu() - all_head_diff[head_to_ablate].flatten().cpu()

make_scatter([(change_in_de_all_heads.flatten().cpu(), "Change in DE of all heads"),
              (change_in_de_all_layers.flatten().cpu(), "Change in DE of all layers"),
              (all_mlp_diff[-1].flatten().cpu(), "Change in DE of last layer"),
              (all_mlp_diff[-2].flatten().cpu(), "Change in DE of second to last layer"),
              (all_mlp_diff[-3].flatten().cpu(), "Change in DE of third to last layer"),
              (change_in_all_heads_except_ablated.cpu(), "Change in DE of all heads EXCEPT ablated head"),]
              , x_data, plot_y_equal_x = True)

# %%
make_scatter([(change_in_all_heads_except_ablated.cpu(), "Change in DE of all heads EXCEPT ablated head"),
              #(change_in_de_all_layers.flatten().cpu(), "Change in DE of all layers"),
              (all_mlp_diff[-1].flatten().cpu(), "Change in DE of last layer")]
              , x_data)
# %%
make_scatter([(ablated_all_layer_direct_effect[-1].flatten().cpu(), "DE of last layer after ablation"),
               ], all_layer_direct_effect[-1].flatten().cpu(), x_label = "DE of last layer", plot_y_equal_x= True, plot_y_equal_minus_x= False)
# %%

significant_indices = torch.where(per_head_direct_effect_flatten > 2)[0]

make_scatter([(change_in_de_all_heads.flatten().cpu(), "Change in DE of all heads"),
              (change_in_de_all_layers.flatten().cpu(), "Change in DE of all layers"),
              (all_mlp_diff[-1].flatten().cpu(), "Change in DE of last layer"),
              (all_mlp_diff[-2].flatten().cpu(), "Change in DE of second to last layer"),
              (all_mlp_diff[-3].flatten().cpu(), "Change in DE of third to last layer"),
              (change_in_all_heads_except_ablated.cpu(), "Change in DE of all heads EXCEPT ablated head"),
              (change_in_de_total, "Change in DE of all heads and layers"),]
              , x_data, plot_y_equal_x = True, indices = significant_indices)

# %%
# "Unflattening" the indices
batch, pos_size = per_head_direct_effect[head_to_ablate].shape


# Convert flattened indices to 2D indices
batch_indices = significant_indices // pos_size
pos_indices = significant_indices % pos_size

# %%
index = 44
# plot the change in DE of all heads on index
show_input(
    all_head_diff[:, :, batch_indices[index], pos_indices[index]],
    all_mlp_diff[:, batch_indices[index], pos_indices[index]],
    title = f"Change in DE of all heads on batch {batch_indices[index]}, pos {pos_indices[index]} | DE = {per_head_direct_effect[head_to_ablate][batch_indices[index], pos_indices[index]]}",
)

# plot original DE
show_input(
    per_head_direct_effect[:, :, batch_indices[index], pos_indices[index]],
    all_layer_direct_effect[:, batch_indices[index], pos_indices[index]],
    title = f"DE on batch {batch_indices[index]}, pos {pos_indices[index]}",
)

# %%


make_scatter([(all_mlp_diff[-1].flatten().cpu(), "Change in DE of last layer")],
             x_data = all_layer_direct_effect[-1].flatten().cpu(),
             x_label = "DE of last layer",
             title = f"DE of last MLP layer vs Change in DE of last MLP layer when ablating {head_to_ablate}",
             indices=significant_indices)
# %%
