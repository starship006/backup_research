"""
We've observed heads with robust self-repression. Let's figure out if there are nice trends here.

I need to explicitly tie this to model iterativity. We should see if the output of these heads is tied to self-repair. And if so, how.


Heads to definitely look at:
 - gpt2small 10.5, which is second highest activating
 - opt 1.3b 21.7, which is third highest activating
 - gpt2medium 22.8, which i found in custom head
 
"""
# %%
import sys
from enum import Enum
# %%
from imports import *
from GOOD_helpers import *
from reused_hooks import overwrite_activation_hook
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    print("Running in notebook mode")
    #%load_ext autoreload
    #%autoreload 2
    model_name = "pythia-160m"
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
batch_size = 80 # 30 -- if model large
prompt_len = 60 # 15 -- if model large

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
# %%
# %% First, see which heads in the model are even useful for predicting these (across ALL positions_)
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)

if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")


# %%
head_to_ablate = (7,8)
model.reset_hooks()
new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = True)
model.reset_hooks()
new_logits = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = False)

FREEZE_FINAL_LN = True

if FREEZE_FINAL_LN:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
else:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False)

all_head_diff: Float[Tensor, "layer head batch pos"] = (ablated_per_head_direct_effect - per_head_direct_effect)
all_mlp_diff: Float[Tensor, "layer batch pos"] = (ablated_all_layer_direct_effect - all_layer_direct_effect)

# %%
def look_for_mean_self_repair():  
    threshold = 0.3
    mask = per_head_direct_effect > threshold # Create a mask where each element is True if it's above the threshold, else False
    selected_elements = torch.where(mask, all_head_diff, 0.0)
    count = mask.sum(dim=(-1, -2)) # Compute the count of elements that contributed to the sum for averaging
    # Handle division by zero in case there are no elements above the threshold in some slices
    count = torch.where(count == 0, torch.tensor(1, device=count.device), count)
    # Compute the average
    thresholded_head_diff = selected_elements.sum(dim=(-1, -2)) / count
    
    
    mask = all_layer_direct_effect > threshold
    selected_elements = torch.where(mask, all_mlp_diff, 0.0)
    count = mask.sum(dim=(-1, -2))  # Summing over 'batch' and 'pos' dimensions
    count = torch.where(count == 0, torch.tensor(1, device=count.device), count)
    thresholded_mlp_diff = selected_elements.sum(dim=(-1, -2)) / count
    
    if in_notebook_mode:
        show_input(thresholded_head_diff,
                thresholded_mlp_diff,
                title = f"Self-repair of heads and MLP Layers upon ablating {head_to_ablate} on threshold {threshold}")

look_for_mean_self_repair()
# %%
def output_self_repair_on_single_top_instance(batch: Union[int, None] = None, pos: Union[int, None] = None,
                                              index_top: Union[int, None] = None):
    if batch is None or pos is None:
        top_indices = topk_of_Nd_tensor(per_head_direct_effect[head_to_ablate[0], head_to_ablate[1]], index_top + 1)
        batch = top_indices[index_top][0]
        pos = top_indices[index_top][1]
        
    assert batch is not None and pos is not None

    index_head_self_repair = (ablated_per_head_direct_effect - per_head_direct_effect)[..., batch, pos]
    index_mlp_self_repair = (ablated_all_layer_direct_effect - all_layer_direct_effect)[..., batch, pos]


    if in_notebook_mode:
            show_input(index_head_self_repair,
                    index_mlp_self_repair,
                    title = f"Self-repair of heads and MLP Layers upon ablating {head_to_ablate} on index {batch, pos}")

    print("Original logit of correct token: ", get_single_correct_logit(logits, batch, pos, owt_tokens))
    print("New logit of correct token: ", get_single_correct_logit(new_logits, batch, pos, owt_tokens))
    print("Difference = ", get_single_correct_logit(new_logits, batch, pos, owt_tokens) - get_single_correct_logit(logits, batch, pos, owt_tokens))
    
    sum = index_head_self_repair.sum() + index_mlp_self_repair.sum()
    print("Value of target head: ", index_head_self_repair[head_to_ablate[0], head_to_ablate[1]])
    print("Sum of all boxes above: ", sum)
    print("LN Scaling old cache", cache['ln_final.hook_scale'][batch, pos])
    print("LN Scaling new cache", new_cache['ln_final.hook_scale'][batch, pos])
    
    print("Sum of all clean DE: ", per_head_direct_effect[..., batch, pos].sum() + all_layer_direct_effect[..., batch, pos].sum())
    print("Sum of all corrupted DE: ", ablated_per_head_direct_effect[..., batch, pos].sum() + ablated_all_layer_direct_effect[..., batch, pos].sum())
    

def measure_self_repair_of_negative_heads(batch: Union[int, None] = None, pos: Union[int, None] = None,
                                              index_top: Union[int, None] = None, verbose = False, negative_threshold = 0):
    
    if batch is None and pos is None and index_top is None:
        top_indices = topk_of_Nd_tensor(per_head_direct_effect[head_to_ablate[0], head_to_ablate[1]], index_top + 1)
        batch = top_indices[index_top][0]
        pos = top_indices[index_top][1]
        
    assert batch is not None and pos is not None
    index_head_self_repair = (ablated_per_head_direct_effect - per_head_direct_effect)[..., batch, pos]
    index_mlp_self_repair = (ablated_all_layer_direct_effect - all_layer_direct_effect)[..., batch, pos]
    
    # get all the negative components
    head_contribution: Float[Tensor, "layer head"] = per_head_direct_effect[..., batch, pos]
    mlp_contribution: Float[Tensor, "layer"] = all_layer_direct_effect[..., batch, pos]
    
    # sum up the negative components
    negative_h_layer_index, negative_h_head_index = torch.where(head_contribution < negative_threshold)
    self_repair_from_negative = 0
    negative_heads = []
    for h_layer, h_head in zip(negative_h_layer_index, negative_h_head_index):
        self_repair_from_negative += index_head_self_repair[h_layer, h_head]
        negative_heads.append((h_layer.item(), h_head.item()))
        
    
    negative_mlp_index = torch.where(mlp_contribution < negative_threshold)[0]
    for mlp_layer in negative_mlp_index:
        self_repair_from_negative += index_mlp_self_repair[mlp_layer]
        
    de_of_original_head = per_head_direct_effect[head_to_ablate[0], head_to_ablate[1], batch, pos] # 0.4
    
    clean_logit = get_single_correct_logit(logits, batch, pos, owt_tokens) # 10
    new_logit = get_single_correct_logit(new_logits, batch, pos, owt_tokens)  # 9.8
    
    expected_change_in_logit = -1 * de_of_original_head # -0.4
    actual_change_in_logit = new_logit - clean_logit # -0.2
    
    
    if verbose:
        print("Diff in logit: ", actual_change_in_logit)
        print("Direct Effect of original head: ", de_of_original_head)
        print("Total self-repair: ", actual_change_in_logit - expected_change_in_logit)
        print("Negative components: ", negative_heads, negative_mlp_index)
        print("Direct Effect change in negative components: ", self_repair_from_negative)
        print("Percent self-repair explained by negative components: ", self_repair_from_negative / (actual_change_in_logit - expected_change_in_logit))
    
    return self_repair_from_negative.item(), (actual_change_in_logit - expected_change_in_logit).item(), de_of_original_head.item()
    

index_top = 0
top_indices = topk_of_Nd_tensor(per_head_direct_effect[head_to_ablate[0], head_to_ablate[1]], index_top + 1)
batch = top_indices[index_top][0]
pos = top_indices[index_top][1]
show_input(per_head_direct_effect[:, :, batch, pos],
           all_layer_direct_effect[:, batch, pos],
           title = f"Direct effect of heads and MLP Layers on index {batch, pos}")   

output_self_repair_on_single_top_instance(batch = batch, pos = pos)
print("-----")
neg_change, self_repair, de_of_head = measure_self_repair_of_negative_heads(batch = batch, pos = pos, verbose=True)
# %%

def get_comp_values_from_function(measurement_function, de_threshold = 1):
    self_repair_values = []
    component_values = []
    de_of_heads = []


    top_indices = topk_of_Nd_tensor(per_head_direct_effect[head_to_ablate[0], head_to_ablate[1]], 1100)
    index = 0
    indices = []

    while True:
        indices.append(index)
        batch = top_indices[index][0]
        pos = top_indices[index][1]
        
        neg_change, self_repair, de_of_head = measurement_function(batch = batch, pos = pos)
        self_repair_values.append(self_repair)
        component_values.append(neg_change)
        de_of_heads.append(de_of_head)
        
        index += 1
        if de_of_head < de_threshold:   
            print("BREAK") 
            break
                    
    other_self_repair_values = []
    other_component_values = []
    other_de_of_heads = []
    other_indices = []

    while index < len(top_indices):
        other_indices.append(index)
        batch = top_indices[index][0]
        pos = top_indices[index][1]
        
        neg_change, self_repair, de_of_head = measurement_function(batch = batch, pos = pos)
        other_self_repair_values.append(self_repair)
        other_component_values.append(neg_change)
        other_de_of_heads.append(de_of_head)
        
        index += 1
        
        
    return self_repair_values, component_values, de_of_heads, other_self_repair_values, other_component_values, other_de_of_heads, indices, other_indices
        
    
# %%
de_threshold = 1
self_repair_values, negative_component_values, de_of_heads, other_self_repair_values, other_negative_component_values, other_de_of_heads, indices, other_indices = get_comp_values_from_function(measure_self_repair_of_negative_heads, de_threshold = de_threshold)

# %%
def create_hover_text(indices, de_values):
    return [f"{i}| DE: {j}" for i, j in zip(indices, de_values)]

def create_scatter_trace(x_values, y_values, de_values, indices, trace_name):
    if indices != None:
        print(indices)
        hover_text = create_hover_text(indices, de_values)
    else:
        hover_text = None
        
    return go.Scatter(
        x=x_values,
        y=y_values,
        mode='markers',
        marker=dict(
            size=3,
            color=de_values,
            colorbar=dict(title="DE of Head 1"),
            colorscale="Viridis",
            cmin=0,
            cmax=3
        ),
        text=hover_text,
        hoverinfo="text+x+y",
        name=trace_name
    )

def create_figure(traces, x_label, y_label, title, head_to_ablate):
    fig = go.Figure()

    for trace in traces:
        fig.add_trace(trace)

    # Determine the range for the y=x line
    all_x_values = [value for trace in traces for value in trace.x]
    min_val, max_val = min(all_x_values), max(all_x_values)

    # Add the y=x line
    line_x = [min_val, max_val]
    line_y = [min_val, max_val]
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='y=x'))

    # Set layout
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        title=title.format(head_to_ablate=head_to_ablate)
    )

    return fig


# %%
trace1 = create_scatter_trace(
    self_repair_values, negative_component_values, 
    de_of_heads, indices, f'DE >= {de_threshold}'
)
trace2 = create_scatter_trace(
    other_self_repair_values, other_negative_component_values, 
    other_de_of_heads, other_indices, f'DE < {de_threshold}'
)
fig = create_figure(
    [trace1, trace2], 
    "Self-Repair = Change in logits", 
    "Change in DE of Negative Components", 
    "Self-repair vs Negative Component Contribution of all instances where {head_to_ablate}",
    head_to_ablate
)
fig.show()
# %% It all appears localised to a MLP layer. Let's check?

def measure_self_repair_in_MLP_layer(batch: Union[int, None] = None, pos: Union[int, None] = None,
                                              index_top: Union[int, None] = None, verbose = False, mlp_layer = -1):
    
    if batch is None and pos is None and index_top is None:
        top_indices = topk_of_Nd_tensor(per_head_direct_effect[head_to_ablate[0], head_to_ablate[1]], index_top + 1)
        batch = top_indices[index_top][0]
        pos = top_indices[index_top][1]
        
    assert batch is not None and pos is not None
    index_head_self_repair = (ablated_per_head_direct_effect - per_head_direct_effect)[..., batch, pos]
    index_mlp_self_repair = (ablated_all_layer_direct_effect - all_layer_direct_effect)[..., batch, pos]
    
    # get all the negative components
    head_contribution: Float[Tensor, "layer head"] = per_head_direct_effect[..., batch, pos]
    mlp_contribution: Float[Tensor, "layer"] = all_layer_direct_effect[..., batch, pos]
    
    # get self-repair of MLP layer
    mlp_layer_self_repair = index_mlp_self_repair[mlp_layer]
    
        
    de_of_original_head = per_head_direct_effect[head_to_ablate[0], head_to_ablate[1], batch, pos] # 0.4
    
    clean_logit = get_single_correct_logit(logits, batch, pos, owt_tokens) # 10
    new_logit = get_single_correct_logit(new_logits, batch, pos, owt_tokens)  # 9.8
    
    expected_change_in_logit = -1 * de_of_original_head # -0.4
    actual_change_in_logit = new_logit - clean_logit # -0.2
    
    return mlp_layer_self_repair.item(), (actual_change_in_logit - expected_change_in_logit).item(), de_of_original_head.item()


# %%
mlp_layer = 11
self_repair_values, specific_component_values, de_of_heads, other_self_repair_values, other_specific_component_values, other_de_of_heads, indices, other_indices = get_comp_values_from_function(partial(measure_self_repair_in_MLP_layer, mlp_layer = mlp_layer))


# %%
trace1 = create_scatter_trace(
    self_repair_values, specific_component_values, 
    de_of_heads, indices, f'DE >= {de_threshold}'
)
trace2 = create_scatter_trace(
    other_self_repair_values, other_specific_component_values, 
    other_de_of_heads, other_indices, f'DE < {de_threshold}'
)
fig = create_figure(
    [trace1, trace2], 
    "Self-Repair = Change in logits", 
    f"Change in DE of MLP {mlp_layer}", 
    f"Total Self-repair vs Self-repair in MLP {mlp_layer}",
    head_to_ablate
)
fig.show()

# %% okay JUST look at the individual values bruh
mlp_DE = all_layer_direct_effect[mlp_layer, :].flatten().cpu()
diff_mlp_DE = all_mlp_diff[mlp_layer, :].flatten().cpu()
de_original_head = per_head_direct_effect[head_to_ablate[0], head_to_ablate[1], :, :].flatten().cpu()



trace1 = create_scatter_trace(
    mlp_DE, diff_mlp_DE, 
    de_original_head, None, f'DE original head'
)

fig = create_figure(
    [trace1], 
    f"DE of MLP {mlp_layer}", 
    f"Self-Repair in MLP {mlp_layer}", 
    f"DE vs Self-repair in MLP {mlp_layer}",
    head_to_ablate
)
fig.show()
# %%
post_ablation_DE = ablated_all_layer_direct_effect[mlp_layer, :].flatten().cpu()

trace1 = create_scatter_trace(
    mlp_DE, post_ablation_DE, 
    de_original_head, None, f'DE original head'
)

fig = create_figure(
    [trace1], 
    f"DE of MLP {mlp_layer}", 
    f"Post-ablation DE of MLP {mlp_layer}", 
    "DE pre- and post- ablation",
    head_to_ablate
)
fig.show()




# %%
def task_vector_influence(scaling: int, batch: int, pos: int):
    new_direct_effects = torch.zeros((model.cfg.n_layers, model.cfg.n_heads)).to(device)
    new_mlp_direct_effects = torch.zeros((model.cfg.n_layers)).to(device)

    # keep the batch to work with functions later
    task_vector: Float[Tensor, "batch d_model"] = cache[utils.get_act_name("result", head_to_ablate[0])][:, pos, head_to_ablate[1], :]


    for layer in range(model.cfg.n_layers):
        model.reset_hooks()
        model.add_hook(utils.get_act_name("resid_pre", layer), partial(add_vector_to_resid, vector = task_vector * scaling, positions = pos))    
        new_logits, new_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()
        
        # measure new direct effects of heads in layer
        new_per_head_direct_effect, new_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=clean_tokens,
                                                                                model = model, display=False)
        
        new_direct_effects[layer] = new_per_head_direct_effect[layer, :, batch, pos]
        new_mlp_direct_effects[layer] = new_all_layer_direct_effect[layer, batch, pos]
        
    return new_direct_effects, new_mlp_direct_effects

scaling = 0
new_direct_effects, new_mlp_direct_effects = task_vector_influence(scaling = scaling, batch = batch, pos = pos)
# %%
show_input(new_direct_effects - per_head_direct_effect[:, :, batch, pos],
           new_mlp_direct_effects - all_layer_direct_effect[:, batch, pos],
           title = f"Diff in head activation when adding {head_to_ablate} output with scaling {scaling} on index {batch, pos}")    
# %%
 





# %% BELOW IS TEST CODE --- ALL FROM OTHER FILES
class output_source(Enum):
    WITH_SAME_AS_RECEIVING_HEAD = 1
    WITH_RANDOM_HEAD = 2
    WITH_RANDOM_DIRECTION = 3
    WITH_FIXED_GLOBAL_TOP_HEAD = 4

class receiving_source(Enum):
    WITH_TOP_HEAD = 1
    WITH_FIXED_GLOBAL_TOP_HEAD = 2
    WITH_CUSTOM_HEAD = 3
    WITH_FIXED_GLOBAL_SECOND_HEAD = 4
    WITH_FIXED_GLOBAL_THIRD_HEAD = 5
    WITH_FIXED_GLOBAL_FOURTH_HEAD = 6




def analyze_constant_head(output_type, receiving_type, scaling, custom_head = None):
    original_de = torch.zeros((batch_size, prompt_len - 1))
    new_de = torch.zeros((batch_size, prompt_len - 1))
    receive_heads = []
    
    if receiving_type == receiving_source.WITH_FIXED_GLOBAL_TOP_HEAD:
        receiving_head = global_top_head
    elif receiving_type == receiving_source.WITH_FIXED_GLOBAL_SECOND_HEAD:
        receiving_head = global_second_head
    elif receiving_type == receiving_source.WITH_FIXED_GLOBAL_THIRD_HEAD:
        receiving_head = global_third_head
    elif receiving_type == receiving_source.WITH_FIXED_GLOBAL_FOURTH_HEAD:
        receiving_head = global_fourth_head
    elif receiving_type == receiving_source.WITH_TOP_HEAD:
        raise Exception("Use analyze_head instead - this function assumes the head is constant")
    elif receiving_type == receiving_source.WITH_CUSTOM_HEAD:
        if custom_head is None:
            raise Exception("Must provide custom_head if receiving_type is WITH_CUSTOM_HEAD")
        receiving_head = custom_head
    else:
        raise Exception("Invalid receiving_type")
    
    
    if output_type == output_source.WITH_RANDOM_HEAD:
        output_head = (torch.randint(0, model.cfg.n_layers, (1,)).item(), torch.randint(0, model.cfg.n_heads, (1,)).item())
        head_out: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", output_head[0])][..., output_head[1], :]
    elif output_type == output_source.WITH_SAME_AS_RECEIVING_HEAD:
        output_head = receiving_head
        head_out: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", output_head[0])][..., output_head[1], :]
    elif output_type == output_source.WITH_RANDOM_DIRECTION:
        temp_output_head = receiving_head
        temp_head_out: Float[Tensor, "d_model"] = cache[utils.get_act_name("result", temp_output_head[0])][-1, -1, temp_output_head[1], :]
        rand_dir =  torch.randn(model.cfg.d_model).to(device)
        head_out: Float[Tensor, "d_model"] = (rand_dir / rand_dir.norm()) * temp_head_out.norm()
        head_out: Float[Tensor, "batch pos d_model"] = einops.repeat(head_out, "d_model -> batch pos d_model", batch = batch_size, pos = prompt_len)
    elif output_type == output_source.WITH_FIXED_GLOBAL_TOP_HEAD:
        head_out: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", global_top_head[0])][..., global_top_head[1], :]
    else:
        raise Exception("Invalid output_type")
    
    head_out = head_out.to(device)
    
    for pos in range(prompt_len - 1):
        pos_head_out: Float[Tensor, "batch d_model"] = head_out[:, pos, :]
        
        model.reset_hooks()
        model.add_hook(utils.get_act_name("resid_pre", receiving_head[0]), partial(add_vector_to_resid, vector = pos_head_out * scaling, positions = pos))
        new_logits, new_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()
        
        # see if the direct effect of the head has changed
        new_per_head_direct_effect, new_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=clean_tokens,
                                                                            model = model, display=False)
        
        original_de[:, pos] = (per_head_direct_effect)[receiving_head[0], receiving_head[1], :, pos]
        new_de[:, pos] = (new_per_head_direct_effect)[receiving_head[0], receiving_head[1], :, pos]
        
        
    # just for consistency rn
    for batch in range(batch_size):
        for pos in range(prompt_len - 1):
            receive_heads.append(receiving_head)
            
            
    return original_de, new_de, receive_heads
# %
# %%
def plot_results(original_de, new_de, receive_heads, scaling, output_type, receiving_type):
    fig = go.Figure()
    
    original_de = original_de.cpu().detach().numpy()
    new_de = new_de.cpu().detach().numpy()
    
    fig.add_trace(
        go.Scatter(
            x=original_de.flatten(), 
            y=new_de.flatten(),
            text="",
            hovertext=receive_heads,
            mode="markers",
            marker=dict(size=2)
        )
    )

    fig.add_trace(
        go.Scatter(x=[original_de.min(), original_de.max()], 
                y=[original_de.min(), original_de.max()], 
                mode='lines', 
                name='y=x')
    )

    fig.update_layout(
        title=f"repression of heads | scaling == {scaling} | {output_type} | {receiving_type}",
        xaxis_title="Original Direct Effect",
        yaxis_title="New Direct Effect"
    )

    return fig

scaling_factors = [-2]
def loop_and_analyze():
    results = {}
    
    
    for output_type in tqdm([output_source.WITH_SAME_AS_RECEIVING_HEAD]):
        receiving_type = receiving_source.WITH_CUSTOM_HEAD
       
        custom_head = head_to_ablate
        for scaling in scaling_factors:
            key = (output_type.name, receiving_type.name + str(custom_head), scaling)
            
            # if key in results:
            #     print("already did this comp")
            #     continue
            
            orig_de, new_de, heads = analyze_constant_head(output_type, receiving_type, scaling = scaling, custom_head=custom_head)
            
            results[key] = (orig_de, new_de, heads)
            plot_results(orig_de, new_de, heads, scaling, output_type, receiving_type).show()

    
         
                
                

loop_and_analyze()

# %%


# %%
# for output_type in tqdm([output_source.WITH_SAME_AS_RECEIVING_HEAD]):
#         for receiving_type in [receiving_source.WITH_FIXED_GLOBAL_TOP_HEAD]:
#             for scaling in [3]:    
#                 print("starting new comp")
#                 orig_de, new_de, heads = analyze_constant_head(output_type, receiving_type, scaling = scaling)
#                 plot_results(orig_de, new_de, heads, scaling, output_type, receiving_type).show()
                
#                 orig_de, new_de, heads = analyze_head(output_type, receiving_type, scaling = scaling)
#                 plot_results(orig_de, new_de, heads, scaling, output_type, receiving_type).show()
                
# %%
