"""
This code explores LN self-repair. It:
- determines how much self-repair is due to LN scales varying
- gets graphs for demonstrating it's importance by freezing LN
"""
# %%
from imports import *

# %%
#%load_ext autoreload
#%autoreload 2
from path_patching import act_patch
from GOOD_helpers import is_notebook, replace_output_hook, replace_model_component_completely, shuffle_owt_tokens_by_batch, get_correct_logit_score, return_item, collect_direct_effect, show_input
#from updated_nmh_dataset_gen import generate_ioi_prompts
# %% Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_notebook_mode = is_notebook()

if in_notebook_mode:
    model_name = "gpt2-small"#"pythia-160m"####
    BATCH_SIZE = 2
    ABLATION_TYPE = "sample" 
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--ablation_type', type=str, default='sample', choices=['mean', 'zero', 'sample'])
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    ABLATION_TYPE = args.ablation_type
assert ABLATION_TYPE in ["mean", "zero", "sample"], "Ablation type must be 'mean', 'zero', or 'sample'."
# %% Import the Model
from transformers import LlamaForCausalLM, LlamaTokenizer
#from constants import LLAMA_MODEL_PATH # change LLAMA_MODEL_PATH to the path of your llama model weights

if "llama" in model_name:
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_MODEL_PATH) 
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<unk>'})
    
    hf_model = LlamaForCausalLM.from_pretrained(LLAMA_MODEL_PATH, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained("llama-7b", hf_model=hf_model, device="cpu", fold_ln=False, center_writing_weights=False, center_unembed=False, tokenizer=tokenizer)
    model: HookedTransformer = model.to("cuda" if torch.cuda.is_available() else "cpu") #type: ignore
else:
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed = True,
        center_writing_weights = True,
        fold_ln = True, # TODO; understand this
        refactor_factored_attn_matrices = False,
        device = device,
    )

safe_model_name = model_name.replace("/", "_")
model.set_use_attn_result(False)
# %% constants
BATCH_SIZE = 10
PROMPT_LEN = 200
# %%
def new_ld_upon_sample_ablation_calc_with_frozen_ln(cache, corrupted_cache, clean_tokens, heads = None, mlp_layers = None):
    """
    Sample ablates and freezes the final LN. Returns new logits
    """
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
    
    

def new_ld_upon_sample_ablation_calc(logits, clean_tokens, corrupted_tokens, heads = None, mlp_layers = None):
    """
    runs activation patching over component and returns new avg_correct_logit_score and avg_ln_scale, averaged over num_runs runs
    
    this is the average logit of the correct token, and new LN scale, upon num_runs sample ablations
    
    logits just used for size
    """
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]


    # Calculate new logit score using act_patch
    avg_logits = act_patch(model, clean_tokens, nodes, return_item, corrupted_tokens, apply_metric_to_cache=False)
    avg_correct_logit_score = get_correct_logit_score(avg_logits, clean_tokens)
    # Calculate new LN
    new_cache = act_patch(model, clean_tokens, nodes, return_item, corrupted_tokens, apply_metric_to_cache=True)
    avg_ln_scale = new_cache['ln_final.hook_scale'][..., 0]

    return avg_correct_logit_score, avg_ln_scale



# %% Generate data
dataset = utils.get_dataset("pile")
dataset_name = "The Pile"


TOTAL_PROMPTS = 300
num_batches = TOTAL_PROMPTS // BATCH_SIZE

new_dataset = utils.tokenize_and_concatenate(dataset, model.tokenizer, max_length = PROMPT_LEN) #type: ignore
all_dataset_tokens = new_dataset['tokens'].to(device) #type: ignore

layer = model.cfg.n_layers - 1 # constant for now
head = 9# constant for now


clean_logit = torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN - 1)) 
change_in_logits = torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN - 1)) #torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN - 1))
change_in_logits_with_frozen_LN =  torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN - 1)) #torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN - 1))
head_direct_effects = torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN - 1))

clean_ln_scale = torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN ))
ablated_ln_scale = torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN ))
# %%
for batch in tqdm(range(num_batches)):
    # Get a batch of clean and corrupted tokens
    clean_batch_offset = batch * BATCH_SIZE
    start_clean_prompt = clean_batch_offset
    end_clean_prompt = clean_batch_offset + BATCH_SIZE
    
    corrupted_batch_offset = (batch + 1) * BATCH_SIZE
    start_corrupted_prompt = corrupted_batch_offset
    end_corrupted_prompt = corrupted_batch_offset + BATCH_SIZE

    clean_tokens = all_dataset_tokens[start_clean_prompt:end_clean_prompt, :PROMPT_LEN]
    corrupted_tokens = all_dataset_tokens[start_corrupted_prompt:end_corrupted_prompt, :PROMPT_LEN]
    assert clean_tokens.shape == corrupted_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
    
    # Run the model on clean/corrupted
    logits, cache = model.run_with_cache(clean_tokens) 
    corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
    
    # Get change in logits + LN scales when sample ablating AND freezing LN
    avg_correct_logit_score, avg_new_LN_scale = new_ld_upon_sample_ablation_calc(logits, clean_tokens, corrupted_tokens, heads = [(layer, head)])
    avg_LN_Frozen_correct_logit_score = new_ld_upon_sample_ablation_calc_with_frozen_ln(cache, corrupted_cache, clean_tokens, heads = [(layer, head)])

    
    
    clean_logit[batch] = get_correct_logit_score(logits, clean_tokens)
    change_in_logits[batch, :] = avg_correct_logit_score - get_correct_logit_score(logits, clean_tokens)
    change_in_logits_with_frozen_LN[batch, :] = avg_LN_Frozen_correct_logit_score - get_correct_logit_score(logits, clean_tokens)
    
    per_head_direct_effect, _ = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False)
    head_direct_effects[batch] = per_head_direct_effect[layer, head]
    
    clean_ln_scale[batch] = cache['ln_final.hook_scale'][..., 0]
    ablated_ln_scale[batch] = avg_new_LN_scale
    
    
# # %%
# per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = in_notebook_mode, collect_individual_neurons = False)
# if in_notebook_mode:
#     show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")

# %% Display Data 
x_data = change_in_logits.flatten(-3, -1)
y_data = change_in_logits_with_frozen_LN.flatten(-3,-1)

color_data = head_direct_effects.flatten(-3,-1).cpu()

# Calculate the 10th and 90th percentiles of color_data
lower_bound = np.percentile(color_data, 0.1)
upper_bound = np.percentile(color_data, 99.9)

# Create scatter plot with color data
scatter_plot = go.Scatter(
    x=x_data, 
    y=y_data, 
    mode='markers', 
    name='Data',
    text=[f'Direct Effect: {val:.2f}' for val in color_data],  # Hover text for each point
    marker=dict(
        color=color_data, 
        colorscale='Viridis',
        cmin=lower_bound,  # Set the lower bound of the color scale
        cmax=upper_bound,  # Set the upper bound of the color scale
        colorbar=dict(title='Direct Effect'),
        showscale=True,
        size = 4
        
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
data = (y_data_filtered / x_data_filtered).cpu().numpy()
title = "Ratio of Change in Logits with Frozen LN to Change in Logits"

# Create a histogram
fig = px.histogram(data, nbins=100, range_x=[-1, 2], title=title)

fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=1,
        x1=1,
        y0=0,
        y1=100,
        line=dict(color="black", dash="dash")
    )
)
# Show the plot
fig.show()

print("Mean: ", (y_data_filtered / x_data_filtered).mean())
# %% 
clean_ln_scale_flat = clean_ln_scale.view(-1).numpy()
ablated_ln_scale_flat = ablated_ln_scale.view(-1).numpy()

# %%
# Create Scatter plot plus y=x line
trace_scatter = go.Scatter(x=clean_ln_scale_flat, y=ablated_ln_scale_flat, mode='markers', 
                          marker=dict(size=4, opacity=1), 
                          name='Clean vs Ablated ln Scale')

trace_identity_line = go.Scatter(x=[min(clean_ln_scale_flat), max(clean_ln_scale_flat)],
                                 y=[min(clean_ln_scale_flat), max(clean_ln_scale_flat)],
                                 mode='lines',
                                 line=dict(color='black', dash='dash'),
                                 name='y=x Line')

# Create layout
layout = go.Layout(title='Scatter Plot of Clean vs Ablated LN Scales',
                   xaxis=dict(title='Clean LN Scale'),
                   yaxis=dict(title='Ablated LN Scale'),
                   )


# Create layout
layout = go.Layout(title='Scatter Plot of Clean vs Ablated LN Scales',
                   xaxis=dict(title='Clean LN Scale'),
                   yaxis=dict(title='Ablated LN Scale'),
                   )

# Create figure
fig = go.Figure(data=[trace_scatter, trace_identity_line], layout=layout)


# Show the figure
fig.show()
# %%
ratio_trace = go.Histogram(x=clean_ln_scale_flat / ablated_ln_scale_flat, name='Clean to Ablated LN Scale Ratio', opacity=1)

# Create layout
layout = go.Layout(title='Histogram of the ratio of Clean to Ablated LN Scale Ratio',
                   xaxis=dict(title='Values'),
                   yaxis=dict(title='Frequency'),
                   barmode='overlay' )


# Create figure
fig = go.Figure(data=[ratio_trace], layout=layout)

fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=1,
        x1=1,
        y0=0,
        y1=5000,
        line=dict(color="black", dash="dash")
    )
)
# Show the figure
fig.show()

print("Percent above 1: ", (clean_ln_scale_flat / ablated_ln_scale_flat > 1).sum() / len(clean_ln_scale_flat))


# %% FILTERED vrsion
clean_ln_scale_flat_filtered, ablated_ln_filtered = filter_top_percentile(clean_ln_scale_flat, ablated_ln_scale_flat, color_data, percentile=2, filter_for_only_positive_ratio = False)
ratio_trace = go.Histogram(x=clean_ln_scale_flat_filtered / ablated_ln_filtered, name='Clean to Ablated LN Scale Ratio', opacity=1)

# Create layout
layout = go.Layout(title='FILTERED Histogram of the ratio of Clean to Ablated LN Scale Ratio',
                   xaxis=dict(title='Values'),
                   yaxis=dict(title='Frequency'),
                   barmode='overlay' )


# Create figure
fig = go.Figure(data=[ratio_trace], layout=layout)

fig.add_shape(
    go.layout.Shape(
        type="line",
        x0=1,
        x1=1,
        y0=0,
        y1=1000,
        line=dict(color="black", dash="dash")
    )
)
# Show the figure
fig.show()

print("Percent above 1: ", (clean_ln_scale_flat_filtered / ablated_ln_filtered > 1).sum() / len(clean_ln_scale_flat_filtered))
# %%
