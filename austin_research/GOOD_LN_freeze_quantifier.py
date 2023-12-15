"""
This code tries to put a number on how much self-repair is explained just by varying LN scales
"""
# %%
from imports import *

# %%
%load_ext autoreload
%autoreload 2
from path_patching import act_patch
from GOOD_helpers import *
from updated_nmh_dataset_gen import generate_ioi_prompts
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "gpt2-small"
else:
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
USE_OWT_DATA = False

if USE_OWT_DATA:
    BATCH_SIZE = 150
    PROMPT_LEN = 30
    owt_dataset = utils.get_dataset("owt")
    owt_dataset_name = "owt"
    
    all_owt_tokens = model.to_tokens(owt_dataset[0:(BATCH_SIZE * 2)]["text"]).to(device)
    owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
    corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
    assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
    
    clean_tokens = owt_tokens
    corrupted_tokens = corrupted_owt_tokens
else:    
    
    double_batch_size = 300
    PROMPTS, _, _ = generate_ioi_prompts(model, double_batch_size)
    
    #ABBA_PROMPTS, BAAB_PROMPTS = PROMPTS[0: int(double_batch_size / 2)], PROMPTS[int(double_batch_size / 2):]
    all_tokens = model.to_tokens(PROMPTS).to(device)


    clean_tokens: Float[Tensor, "batch pos"] = all_tokens[0:int(double_batch_size / 2)]
    corrupted_tokens = all_tokens[int(double_batch_size / 2):]
    
    assert clean_tokens.shape == corrupted_tokens.shape 
    BATCH_SIZE = clean_tokens.shape[0]
    PROMPT_LEN = clean_tokens.shape[1]
# %%
logits, cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
# %%
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = in_notebook_mode, collect_individual_neurons = False)
if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")
# %%
def new_ld_upon_sample_ablation_calc_with_frozen_ln(heads = None, mlp_layers = None):
    assert heads != None and len(heads) == 1 and mlp_layers == None
    
    model.reset_hooks()
    corrupted_output: Float["batch pos d_head"] = corrupted_cache[utils.get_act_name("z",  heads[0][0])][...,  heads[0][1], :]
    model.add_hook(utils.get_act_name("z", heads[0][0]), partial(replace_output_hook, new_output = corrupted_output, head = heads[0][1]))
    model.add_hook('ln_final.hook_scale', partial(replace_model_component_completely, new_model_comp = cache['ln_final.hook_scale']))
    ablated_logits = model(clean_tokens)
    model.reset_hooks()
    # get change in direct effect compared to original
    avg_correct_logit_score = get_correct_logit_score(ablated_logits, clean_tokens)
    return avg_correct_logit_score
    

def new_ld_upon_sample_ablation_calc(heads = None, mlp_layers = None, num_runs = 2):
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
        # Shuffle clean_tokens by batch
        shuffled_corrupted_tokens = shuffle_owt_tokens_by_batch(corrupted_tokens)
        # Calculate new_logits using act_patch
        new_logits = act_patch(model, clean_tokens, nodes, return_item, shuffled_corrupted_tokens, apply_metric_to_cache=False)
        logits_accumulator += new_logits

    avg_logits = logits_accumulator / num_runs
    # get change in direct effect compared to original
    avg_correct_logit_score = get_correct_logit_score(avg_logits, clean_tokens)
    return avg_correct_logit_score




# %%
change_in_logits = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN - 1))
change_in_logits_with_frozen_LN = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN - 1))

for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        avg_correct_logit_score = new_ld_upon_sample_ablation_calc(heads = [(layer, head)])
        avg_LN_Frozen_correct_logit_score = new_ld_upon_sample_ablation_calc_with_frozen_ln(heads = [(layer, head)])
        
        change_in_logits[layer, head] = avg_correct_logit_score - get_correct_logit_score(logits, clean_tokens)
        change_in_logits_with_frozen_LN[layer, head] = avg_LN_Frozen_correct_logit_score - get_correct_logit_score(logits, clean_tokens)
                

# %%
layer = 10
head = 0
x_data = change_in_logits.flatten(-2, -1)[layer, head]
y_data = change_in_logits_with_frozen_LN.flatten(-2,-1)[layer, head]
color_data = per_head_direct_effect[layer, head].flatten(-2,-1).cpu()

# Calculate the 10th and 90th percentiles of color_data
lower_bound = np.percentile(color_data, 1)
upper_bound = np.percentile(color_data, 99)

# Create scatter plot with color data
scatter_plot = go.Scatter(
    x=x_data, 
    y=y_data, 
    mode='markers', 
    name='Data',
    text=[f'Color Value: {val:.2f}' for val in color_data],  # Hover text for each point
    marker=dict(
        color=color_data, 
        colorscale='Viridis',
        cmin=lower_bound,  # Set the lower bound of the color scale
        cmax=upper_bound,  # Set the upper bound of the color scale
        colorbar=dict(title='Direct Effect'),
        showscale=True
    )
)
# Create y=x line
line_range = [min(min(x_data), min(y_data)), max(max(x_data), max(y_data))]
line_plot = go.Scatter(x=line_range, y=line_range, mode='lines', name='y=x')

# Combine plots
fig = go.Figure(data=[scatter_plot, line_plot])

# Update layout
fig.update_layout(
    title='Scatter Plot with y=x Line', 
    xaxis_title='Change in Logits', 
    yaxis_title='Change in Logits with Frozen LN',
    yaxis=dict(scaleanchor="x", scaleratio=1)
)

# Show plot
fig.show()
# %%

def filter_top_percentile(x_data, y_data, color_data, percentile = 10, filter_for_only_positive_ratio = False):
    # Calculate the percentile'tj of color_data
    percentile_data = np.percentile(color_data, 100 - percentile)
    print("Percentile: ", percentile_data)

    # Filter indices where color_data is above the 90th percentile
    if filter_for_only_positive_ratio:
        top_indices = np.where((color_data >= percentile_data) & (y_data / x_data > 0))[0]
    else:
        top_indices = np.where(color_data >= percentile_data)[0]

    # Filter x_data and y_data based on these indices
    x_data_filtered = x_data[top_indices]
    y_data_filtered = y_data[top_indices]

    return x_data_filtered, y_data_filtered



x_data_filtered, y_data_filtered = filter_top_percentile(x_data, y_data, color_data, percentile=1, filter_for_only_positive_ratio = False)

# %%
#x_data_filtered, y_data_filtered = x_data, y_data
histogram(y_data_filtered / x_data_filtered, title = "Ratio of Change in Logits with Frozen LN to Change in Logits", nbins = 50)

print("Mean: ", (y_data_filtered / x_data_filtered).mean())
# %%
