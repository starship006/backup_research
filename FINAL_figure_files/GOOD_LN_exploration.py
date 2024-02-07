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
from GOOD_helpers import is_notebook, replace_output_hook, replace_model_component_completely, shuffle_owt_tokens_by_batch, get_correct_logit_score, return_item, collect_direct_effect, show_input, prepare_dataset
#from updated_nmh_dataset_gen import generate_ioi_prompts
# %% Constants

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_notebook_mode = is_notebook()

if in_notebook_mode:
    model_name = "gpt2-small"#""####
    BATCH_SIZE = 12
    #ablation_type = "sample" 
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--ablation_type', type=str, default='sample', choices=['mean', 'zero', 'sample'])
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    #ablation_type = args.ablation_type
#assert ablation_type in ["sample"], "Ablation type must be 'sample', currently;  'mean', 'zero' not yet supported."

PROMPT_LEN = 200

# %% Import the Model
from transformers import LlamaForCausalLM, LlamaTokenizer
from constants import LLAMA_MODEL_PATH # change LLAMA_MODEL_PATH to the path of your llama model weights

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

# %%
# def new_ld_upon_sample_ablation_calc_with_frozen_ln(cache, corrupted_cache, clean_tokens, heads = None, mlp_layers = None):
#     """
#     Sample ablates and freezes the final LN. Returns new logits
#     """
#     assert heads != None and len(heads) == 1 and mlp_layers == None
    
#     model.reset_hooks()
#     corrupted_output: Float["batch pos d_head"] = corrupted_cache[utils.get_act_name("z",  heads[0][0])][...,  heads[0][1], :]
#     model.add_hook(utils.get_act_name("z", heads[0][0]), partial(replace_output_hook, new_output = corrupted_output, head = heads[0][1]))
#     model.add_hook('ln_final.hook_scale', partial(replace_model_component_completely, new_model_comp = cache['ln_final.hook_scale']))
#     ablated_logits = model(clean_tokens)
#     model.reset_hooks()
#     # get change in direct effect compared to original
#     avg_correct_logit_score = get_correct_logit_score(ablated_logits, clean_tokens)
#     return avg_correct_logit_score
    
    

def new_ld_upon_sample_ablation_calc(logits, clean_tokens, corrupted_tokens, clean_cache, ablation_type, heads = None, mlp_layers = None):
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


    if ablation_type == "zero":
        new_logits = act_patch(model, clean_tokens, nodes, return_item, new_cache = "zero", apply_metric_to_cache=False)# type: ignore
        new_cache = act_patch(model, clean_tokens, nodes, return_item, new_cache = "zero", apply_metric_to_cache=True)# type: ignore
    elif ablation_type == "mean":
        assert clean_cache is not None, "clean_cache must be provided for mean ablation"
        assert len(nodes) == 1 and mlp_layers == None and heads != None, "mean ablation currently only works for one head"
        
        # get average output of layer
        avg_output_of_layer = clean_cache[utils.get_act_name("z", heads[0][0])][:, :, heads[0][1], :].mean((0,1))
        #W_U = model.W_O[heads[0][0], heads[0][1]]
        #avg_output_of_layer = einops.einsum(avg_output_of_layer, W_U, "d_head, d_head d_model -> d_model")
        avg_output_of_layer = einops.repeat(avg_output_of_layer, "d_model -> batch seq d_model", batch = clean_tokens.shape[0], seq = clean_tokens.shape[1])
        
        # run with hook which mean ablates
        model.reset_hooks()
        hook = partial(replace_output_hook, new_output = avg_output_of_layer, head = heads[0][1])
        model.add_hook(utils.get_act_name("z", heads[0][0]), hook)
        new_logits, new_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()
    else:
        new_logits = act_patch(model, clean_tokens, nodes, return_item, corrupted_tokens, apply_metric_to_cache=False)
        new_cache = act_patch(model, clean_tokens, nodes, return_item, corrupted_tokens, apply_metric_to_cache=True)

    assert isinstance(new_cache, ActivationCache)
    assert isinstance(new_logits, torch.Tensor)
    
    # Calculate new logit score using act_patch
    avg_correct_logit_score = get_correct_logit_score(new_logits, clean_tokens)
    # Calculate new LN
    avg_ln_scale = new_cache['ln_final.hook_scale'][..., 0]

    return avg_correct_logit_score, avg_ln_scale



# %% Generate data
TOTAL_TOKENS = ((1_000_000 // (PROMPT_LEN * BATCH_SIZE)) + 1) * (PROMPT_LEN * BATCH_SIZE)
dataset, num_batches = prepare_dataset(model, device, TOTAL_TOKENS, BATCH_SIZE, PROMPT_LEN, False, "pile")

layer = model.cfg.n_layers - 1 # constant for now
head = 11# constant for now : 2 was main 

ablation_types = ["sample", "mean", "zero"]


#change_in_logits_with_frozen_LN =  torch.zeros((num_batches, BATCH_SIZE, PROMPT_LEN - 1)) #torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN - 1))

clean_logit = torch.zeros((len(ablation_types), num_batches, BATCH_SIZE, PROMPT_LEN - 1)) 
change_in_logits = torch.zeros((len(ablation_types), num_batches, BATCH_SIZE, PROMPT_LEN - 1)) #torch.zeros((model.cfg.n_layers, model.cfg.n_heads, BATCH_SIZE, PROMPT_LEN - 1))
head_direct_effects = torch.zeros((len(ablation_types), num_batches, BATCH_SIZE, PROMPT_LEN - 1))
clean_ln_scale = torch.zeros((len(ablation_types), num_batches, BATCH_SIZE, PROMPT_LEN ))
ablated_ln_scale = torch.zeros((len(ablation_types), num_batches, BATCH_SIZE, PROMPT_LEN ))
# %%
pbar = tqdm(total=num_batches, desc='Processing batches')

for batch_idx, clean_tokens, corrupted_tokens in dataset:
    for i, ablation_type in enumerate(ablation_types):
        assert clean_tokens.shape == corrupted_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
        
        # Run the model on clean/corrupted
        logits, cache = model.run_with_cache(clean_tokens) 
        corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
        
        # Get change in logits + LN scales when sample ablating AND freezing LN
        avg_correct_logit_score, avg_new_LN_scale = new_ld_upon_sample_ablation_calc(logits, clean_tokens, corrupted_tokens, cache, ablation_type, heads = [(layer, head)])
        #avg_LN_Frozen_correct_logit_score = new_ld_upon_sample_ablation_calc_with_frozen_ln(cache, corrupted_cache, clean_tokens, heads = [(layer, head)])

        
        clean_logit[i, batch_idx] = get_correct_logit_score(logits, clean_tokens)
        change_in_logits[i, batch_idx, :] = avg_correct_logit_score - get_correct_logit_score(logits, clean_tokens)
        #change_in_logits_with_frozen_LN[batch_idx, :] = avg_LN_Frozen_correct_logit_score - get_correct_logit_score(logits, clean_tokens)
        
        per_head_direct_effect, _ = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False)
        head_direct_effects[i, batch_idx] = per_head_direct_effect[layer, head]
        
        clean_ln_scale[i, batch_idx] = cache['ln_final.hook_scale'][..., 0]
        ablated_ln_scale[i, batch_idx] = avg_new_LN_scale
        
    pbar.update(1)
    
pbar.close()

# # %%
# per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = in_notebook_mode, collect_individual_neurons = False)
# if in_notebook_mode:
#     show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")

# %%
def filter_top_percentile(x_data, y_data, color_data, percentile = 10, filter_for_only_positive_ratio = False):
    # Calculate the percentile'tj of color_data
    percentile_data = np.percentile(color_data, 100 - percentile)
    print("Percentile equal to:", percentile_data)

    # Filter indices where color_data is above the 90th percentile
    if filter_for_only_positive_ratio:
        top_indices = np.where((color_data > percentile_data) & (y_data / x_data > 0))[0]
    else:
        top_indices = np.where(color_data > percentile_data)[0]

    print(np.where(color_data > percentile_data))
    # Filter x_data and y_data based on these indices
    x_data_filtered = x_data[top_indices]
    y_data_filtered = y_data[top_indices]

    return x_data_filtered, y_data_filtered
# # %% Display Data  for the LayerNorm Freezing Technique!
# x_data = change_in_logits.flatten(-3, -1)
# y_data = change_in_logits_with_frozen_LN.flatten(-3,-1)

# color_data = head_direct_effects.flatten(-3,-1).cpu()

# # Calculate the 10th and 90th percentiles of color_data
# lower_bound = np.percentile(color_data, 0.1)
# upper_bound = np.percentile(color_data, 99.9)

# # Create scatter plot with color data
# scatter_plot = go.Scatter(
#     x=x_data, 
#     y=y_data, 
#     mode='markers', 
#     name='Data',
#     text=[f'Direct Effect: {val:.2f}' for val in color_data],  # Hover text for each point
#     marker=dict(
#         color=color_data, 
#         colorscale='Viridis',
#         cmin=lower_bound,  # Set the lower bound of the color scale
#         cmax=upper_bound,  # Set the upper bound of the color scale
#         colorbar=dict(title='Direct Effect'),
#         showscale=True,
#         size = 4
        
#     )
# )
# # Create y=x line
# line_range = [min(min(x_data), min(y_data)), max(max(x_data), max(y_data))]
# line_plot = go.Scatter(x=line_range, y=line_range, mode='lines', name='y=x')

# # Combine plots
# fig = go.Figure(data=[scatter_plot, line_plot])

# # Update layout
# fig.update_layout(
#     title='Scatter Plot with y=x Line', 
#     xaxis_title='Change in Logits', 
#     yaxis_title='Change in Logits with Frozen LN',
#     yaxis=dict(scaleanchor="x", scaleratio=1)
# )

# # Show plot
# if in_notebook_mode:
#     fig.show()


# %%
# x_data_filtered, y_data_filtered = filter_top_percentile(x_data, y_data, color_data, percentile=2, filter_for_only_positive_ratio = False)

# # %%
# #x_data_filtered, y_data_filtered = x_data, y_data
# data = (y_data_filtered / x_data_filtered).cpu().numpy()
# title = "Ratio of Change in Logits with Frozen LN to Change in Logits"

# # Create a histogram
# fig = px.histogram(data, nbins=200001, range_x=[-1, 2], title=title)

# fig.add_shape(
#     go.layout.Shape(
#         type="line",
#         x0=1,
#         x1=1,
#         y0=0,
#         y1=100,
#         line=dict(color="black", dash="dash")
#     )
# )
# # Show the plot
# if in_notebook_mode:
#     fig.show()

# print("Mean: ", (y_data_filtered / x_data_filtered).mean())
# # %% 
clean_ln_scale_flat = clean_ln_scale[..., :-1].flatten(-3,-1).cpu()
ablated_ln_scale_flat = ablated_ln_scale[..., :-1].flatten(-3,-1).cpu()
color_data = head_direct_effects.flatten(-3,-1).cpu()

assert clean_ln_scale_flat.shape == ablated_ln_scale_flat.shape == color_data.shape
# assert clean_ln_scale_flat.shape == ablated_ln_scale_flat.shape == color_data.shape
# # %%
# # Create Scatter plot plus y=x line
# trace_scatter = go.Scatter(x=clean_ln_scale_flat, y=ablated_ln_scale_flat, mode='markers', 
#                           marker=dict(size=4, opacity=1), 
#                           name='Clean vs Ablated ln Scale')

# trace_identity_line = go.Scatter(x=[min(clean_ln_scale_flat), max(clean_ln_scale_flat)],
#                                  y=[min(clean_ln_scale_flat), max(clean_ln_scale_flat)],
#                                  mode='lines',
#                                  line=dict(color='black', dash='dash'),
#                                  name='y=x Line')

# # Create layout
# layout = go.Layout(title='Scatter Plot of Clean vs Ablated LN Scales',
#                    xaxis=dict(title='Clean LN Scale'),
#                    yaxis=dict(title='Ablated LN Scale'),
#                    )


# # Create layout
# layout = go.Layout(title='Scatter Plot of Clean vs Ablated LN Scales',
#                    xaxis=dict(title='Clean LN Scale'),
#                    yaxis=dict(title='Ablated LN Scale'),
#                    )

# # Create figure
# fig = go.Figure(data=[trace_scatter, trace_identity_line], layout=layout)


# # Show the figure
# if in_notebook_mode:
#     fig.show()
# %%
  # Set the desired size of each histogram bin here

def plot_graph(cleans, ablateds, box_size=0.0001):
    fig = go.Figure()
    
    # Define a tropical color scale
    colors = ["#1515DC", "#FF4365", "#61D095"]
    
    correct_ablation_types = ['resample', 'mean', 'zero']
    
    for i, ablation_type in enumerate(correct_ablation_types):
        ratio_trace = go.Histogram(
            x=cleans[i] / ablateds[i],
            name=f'{ablation_type}',
            opacity=0.6,  # Adjust the opacity as needed (0.0 to 1.0)
            xbins=dict(size=box_size),
            marker_color=colors[i],  # Use modulo to repeat colors
        )
        
        fig.add_trace(ratio_trace)
        
    layout = go.Layout(
        #title='Histogram of the ratio of Clean to Ablated LN Scale Ratio',
        xaxis=dict(title='Clean to Ablated LN Scale Ratio'),
        yaxis=dict(title='Frequency'),
        barmode='overlay',
        width=700,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    fig.update_xaxes(linecolor='black')
    fig.update_yaxes(linecolor='black')
    
    fig.update_layout(layout)
    fig.add_shape(
        go.layout.Shape(
            type="line",
            x0=1,
            x1=1,
            y0=0,
            y1=1100,
            line=dict(color="black", dash="dash")
        )
    )
    
        
    return fig
    
# %% wait dont actually run this it takes forever unless you use a smaller batch size
# plot_graph(clean_ln_scale_flat, ablated_ln_scale_flat)
    

# %%
#fig.write_image(f"figures/LN_graphs/{safe_model_name}_L{layer}H{head}_everything_.pdf")
for i in range(len(ablation_types)):
    print(f"Percent above 1: ", (clean_ln_scale_flat[i] / ablated_ln_scale_flat[i] > 1).sum() / len(clean_ln_scale_flat[i]))



# %% FILTERED vrsion
filter_percentile = 2
clean_ln_scale_flat_filtered = []
ablated_ln_filtered = []

for i in range(len(ablation_types)):
    a, b = filter_top_percentile(clean_ln_scale_flat[i], ablated_ln_scale_flat[i], color_data[i], percentile=filter_percentile, filter_for_only_positive_ratio = False)
    clean_ln_scale_flat_filtered.append(a)
    ablated_ln_filtered.append(b)
    
    
clean_ln_scale_flat_filtered = torch.stack(clean_ln_scale_flat_filtered)
ablated_ln_filtered = torch.stack(ablated_ln_filtered)
# %%
fig = plot_graph(clean_ln_scale_flat_filtered, ablated_ln_filtered, box_size = 0.0005)
fig.show()
fig.write_image(f"figures/LN_graphs/{safe_model_name}_L{layer}H{head}_top{filter_percentile}_percentile_LINE_.pdf")
# %%

for i in range(len(ablation_types)):
    print(f"Percent above 1: ", (clean_ln_scale_flat_filtered[i] / ablated_ln_filtered[i] > 1).sum() / len(clean_ln_scale_flat_filtered[i]))




# %%
# filter_top_percentile(clean_ln_scale_flat, ablated_ln_scale_flat, color_data, percentile=filter_percentile, filter_for_only_positive_ratio = False)
# ratio_trace = go.Histogram(x=clean_ln_scale_flat_filtered / ablated_ln_filtered, name='Clean to Ablated LN Scale Ratio', opacity=1)

# # Create layout
# layout = go.Layout(title=f"Clean to Ablated LN Scale Ratio, on top {filter_percentile}% of {safe_model_name} L{layer}H{head} Direct Effect",
#                    xaxis=dict(title='Values'),
#                    yaxis=dict(title='Frequency'),
#                    barmode='overlay' )


# # Create figure
# fig = go.Figure(data=[ratio_trace], layout=layout)


# # Show the figure
# if in_notebook_mode:
#     fig.show()
    
    
# print(f"Percent above 1: ", (clean_ln_scale_flat_filtered / ablated_ln_filtered > 1).sum() / len(clean_ln_scale_flat_filtered))
# fig.write_image(f"figures/LN_graphs/{safe_model_name}_L{layer}H{head}_top{filter_percentile}_percentile_.pdf")
# # %%


# fig.add_shape(
#     go.layout.Shape(
#         type="line",
#         x0=1,
#         x1=1,
#         y0=0,
#         y1=len(clean_ln_scale_flat_filtered) / 20,
#         line=dict(color="black", dash="dash")
#     )
# )
# if in_notebook_mode:
#     fig.show()
# fig.write_image(f"figures/LN_graphs/{safe_model_name}_L{layer}H{head}_top{filter_percentile}_percentile_LINE_.pdf")
# # %%
