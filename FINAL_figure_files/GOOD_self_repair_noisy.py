"""
This code generates graphs that display how noisy self-repair is; i.e., how many prompts aren't self-repaired
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, show_input, collect_direct_effect, get_single_correct_logit, topk_of_Nd_tensor, return_item, get_correct_logit_score, prepare_dataset
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/breakdown_self_repair/"


if in_notebook_mode:
    model_name = "pythia-410m"
    BATCH_SIZE = 30
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)  
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size


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
dataset = utils.get_dataset("pile")
dataset_name = "The Pile"
all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
# %% 
def ablate_instance_and_get_repair(head: tuple, clean_tokens: Tensor, corrupted_tokens: Tensor,
                                           per_head_direct_effect: Union[Tensor, None] = None,
                                           all_layer_direct_effect: Union[Tensor, None] = None,
                                           
                                           cache: Union[ActivationCache, None] = None,
                                           logits: Union[Tensor, None] = None,):    
    
    if cache is None or logits is None:
        print("Cache not provided")
        cache, logits = model.run_with_cache(clean_tokens)
        assert cache != None
        assert logits != None
        
        
    if per_head_direct_effect is None or all_layer_direct_effect is None:
        print("Per head direct effect not provided")
        per_head_direct_effect, all_layer_direct_effect, per_neuron_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = True, cache_for_scaling = cache)
        assert per_head_direct_effect != None
        assert all_layer_direct_effect != None
    
    # Run ablation and get new cache/logit/DEs
    ablated_cache: ActivationCache = act_patch(model, clean_tokens, [Node("z", head[0], head[1])], return_item, corrupted_tokens, apply_metric_to_cache = True) #type: ignore
    ablated_logits = act_patch(model, clean_tokens, [Node("z", head[0], head[1])], return_item, corrupted_tokens, apply_metric_to_cache = False)
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(ablated_cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
    
    # get the logit difference between everything
    correct_logits = get_correct_logit_score(logits, clean_tokens)
    ablated_logits = get_correct_logit_score(ablated_logits, clean_tokens)
    logit_diffs = ablated_logits - correct_logits
    
    # Get Direct Effect of Heads Pre/Post-Ablation
    direct_effects = per_head_direct_effect[head[0],head[1]] 
    ablated_direct_effects = ablated_per_head_direct_effect[head[0],head[1]]
    
    
    # Calculate self-repair values
    self_repair = logit_diffs - (ablated_direct_effects - direct_effects)
    self_repair_from_heads = (ablated_per_head_direct_effect - per_head_direct_effect).sum((0,1)) - (ablated_per_head_direct_effect - per_head_direct_effect)[head[0],head[1]]
    self_repair_from_layers = (ablated_all_layer_direct_effect - all_layer_direct_effect).sum(0)
    self_repair_from_LN = self_repair - self_repair_from_heads - self_repair_from_layers   # "self repair from LN" is the residual
    
    return logit_diffs, direct_effects, ablated_direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN, self_repair
        
# %% We need to iterate through the dataset to find the 
PROMPT_LEN = 100
TOTAL_TOKENS = ((5_000 // (PROMPT_LEN * BATCH_SIZE)) + 1) * (PROMPT_LEN * BATCH_SIZE)
dataset, num_batches = prepare_dataset(model, device, TOTAL_TOKENS, BATCH_SIZE, PROMPT_LEN, False, "pile")

TOTAL_PROMPTS_TO_ITERATE_THROUGH = num_batches * BATCH_SIZE
#TOTAL_PROMPTS_TO_ITERATE_THROUGH = 300#3000
#PROMPT_LEN = 100#400

# %%
logit_diffs_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
direct_effects_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
# ablated_direct_effects_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
# self_repair_from_heads_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
# self_repair_from_layers_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
self_repair_from_LN_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
self_repair_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
# %%
pbar = tqdm(total=num_batches, desc='Processing batches')


ablate_heads = [(22, 11), 
                (20, 6),
                (23, 1),
                (20,14)] 


#ablate_heads = [(model.cfg.n_layers - 1, i) for i in range(6)]
for batch_idx, clean_tokens, corrupted_tokens in dataset:
    assert clean_tokens.shape == corrupted_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
    
    # Cache clean/corrupted model activations + direct effects
    logits, cache = model.run_with_cache(clean_tokens)
    start_clean_prompt = batch_idx * BATCH_SIZE
    end_clean_prompt = start_clean_prompt + BATCH_SIZE
    
    for ablate_layer, ablate_head in ablate_heads:
        per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False)
        logit_diffs, direct_effects, _, _, _, self_repair_from_LN, self_repair = ablate_instance_and_get_repair((ablate_layer, ablate_head), clean_tokens, corrupted_tokens, per_head_direct_effect, all_layer_direct_effect, cache, logits)
        
        logit_diffs_across_everything[start_clean_prompt:end_clean_prompt, :, ablate_layer, ablate_head] = logit_diffs
        direct_effects_across_everything[start_clean_prompt:end_clean_prompt, :, ablate_layer, ablate_head] = direct_effects
        self_repair_from_LN_across_everything[start_clean_prompt:end_clean_prompt, :, ablate_layer, ablate_head] = self_repair_from_LN
        self_repair_across_everything[start_clean_prompt:end_clean_prompt, :, ablate_layer, ablate_head] = self_repair
    
    pbar.update(1)
# %% Self-repair is noisy: just plot DE vs change in logits

cols = 2

fig = plotly.subplots.make_subplots(
    rows=2, cols=cols,
    shared_xaxes=False,
    shared_yaxes=True,
    vertical_spacing=0.25,
    horizontal_spacing=0.15,
    specs=[[{"type": "scatter"}] * cols]*2,
    subplot_titles=[f"L{layer}H{head}" for layer, head in ablate_heads],
)

colors = ['#668F80', '#4A6670', '#D6A2AD', '#A0AF84']

for i, (layer, head) in enumerate(ablate_heads):
    logit_diffs = logit_diffs_across_everything[:, :, layer, head].flatten().cpu().numpy()
    direct_effect = direct_effects_across_everything[:, :, layer, head].flatten().cpu().numpy()

    fig.add_trace(go.Scatter(
        x=direct_effect,
        y=logit_diffs,
        mode='markers',
        name=f"L{layer}H{head}",
        marker=dict(size=2, color=colors[i]),
    ), row=(i // cols) + 1, col=(i % cols) + 1)
    
    fig.add_shape(
        go.layout.Shape(x3
            type='line',
            x0=-4,
            y0=4,
            x1=4,
            y1=-4,
            line=dict(color='grey', dash='dash'),
        ),
        row=(i // cols) + 1,
        col=(i % cols) + 1
    )
    fig.update_xaxes(linecolor='#C3B59F', row=(i // cols) + 1, col=(i % cols) + 1)
    fig.update_yaxes(linecolor='#C3B59F', row=(i // cols) + 1, col=(i % cols) + 1)


fig.update_layout(
    title=f"Direct Effect vs. Change in Logits when Ablating Attention Heads in {model_name}",
    showlegend=False,
    #margin=dict(l=40, r=40, t=60, b=40),
    width=550,
    height=550,
    plot_bgcolor='white',  # Add this line to set the background color to white
)

fig.update_xaxes(title_text="Direct Effect", zeroline=True, zerolinecolor='black', range=[-4, 4])
fig.update_yaxes(title_text="Logit Difference", zeroline = True, zerolinecolor='black', range=[-4, 4])
fig.show()
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"many_tokens_direct_effect_vs_logit_diff_{safe_model_name}.pdf")



# %% First: Self-repair is noisy. Plot the distribution of self-repair values across all prompts
fig = go.Figure()

min_val = self_repair_across_everything.mean(1).min()
max_val = self_repair_across_everything.mean(1).max()

num_bins = 500
bin_size = (max_val - min_val) / num_bins


for (layer, head) in ablate_heads:
    # Compute the mean self-repair across the "everything" dimension for the specific layer and head
    data = self_repair_across_everything[:, :, layer, head].mean(1).cpu().numpy()

    # Add a histogram to the figure for this layer-head pair
    fig.add_trace(go.Histogram(
        x=data,
        name=f"L{layer}H{head}",  # Naming each trace with the corresponding layer-head pair
        opacity=0.75,  # Adjust opacity to make the overlays visible,
        nbinsx=int(np.ceil((data.max() - data.min()) / bin_size))
    ))
    
fig.update_layout(
        title=f"Mean Self-Repair on Prompt when ablating various layers and heads | {model_name}",
        xaxis_title="Mean Self-Repair on Prompt",
        yaxis_title="Count",
        # Additional layout parameters for better visualization
        barmode='overlay'
    )

fig.show()

# %% Next: Self-repair is noisy, but coorelates with the DE of the head
fig = go.Figure()

# Loop over each (layer, head) tuple
for (layer, head) in ablate_heads:
    # Calculate the mean self-repair and direct effect for this layer-head pair
    self_repair = self_repair_across_everything[:10, :, layer, head].flatten().cpu().numpy()
    direct_effect = direct_effects_across_everything[:10, :, layer, head].flatten().cpu().numpy()

    # Add a scatter trace to the figure for this layer-head pair
    fig.add_trace(go.Scatter(
        x=direct_effect,
        y=self_repair,
        mode='markers',  # Change to 'lines+markers' if you prefer
        name=f"L{layer}H{head}",  # Naming each trace with the corresponding layer-head pair,
        marker=dict(size=2)  # Decreased marker size
    ))


fig.update_layout(
        title=f"Direct Effect vs. Self-Repair when ablating various layers and heads | {model_name}",
        xaxis_title="Mean Direct Effect on Prompt",
        yaxis_title="Mean Self-Repair on Prompt",
        legend_title="Attention Head"
    )

fig.show()
#fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"direct_effect_vs_self_repair_{safe_model_name}.html")

# %% Especially the LN part
# fig = go.Figure()

# # Loop over each (layer, head) tuple
# for (layer, head) in ablate_heads:
#     # Calculate the mean self-repair and direct effect for this layer-head pair
#     ln_self_repair = self_repair_from_LN_across_everything[:10, :, layer, head].flatten().cpu().numpy()
#     direct_effect = direct_effects_across_everything[:10, :, layer, head].flatten().cpu().numpy()

#     # Add a scatter trace to the figure for this layer-head pair
#     fig.add_trace(go.Scatter(
#         x=direct_effect,
#         y=ln_self_repair,
#         mode='markers',  # Change to 'lines+markers' if you prefer
#         name=f"L{layer}H{head}",  # Naming each trace with the corresponding layer-head pair,
#         marker=dict(size=2)  # Decreased marker size
#     ))


# fig.update_layout(
#         title=f"Direct Effect vs. LN Self-Repair when ablating various layers and heads | {model_name}",
#         xaxis_title="Mean Direct Effect on Prompt",
#         yaxis_title="Mean Self-Repair on Prompt",
#         legend_title="Attention Head"
#     )

# fig.show()



# %%
# # %% Save the data to pickle
# tensors_to_save = {
#     "condensed_logit_diff": condensed_logit_diff,
#     "condensed_direct_effects": condensed_direct_effects,
#     "condensed_ablated_direct_effects": condensed_ablated_direct_effects,
#     "condensed_self_repair_from_heads": condensed_self_repair_from_heads,
#     "condensed_self_repair_from_layers": condensed_self_repair_from_layers,
#     "condensed_self_repair_from_LN": condensed_self_repair_from_LN
# }

# # Loop through the dictionary and save each tensor
# for tensor_name, tensor_data in tensors_to_save.items():
#     with open(FOLDER_TO_STORE_PICKLES + f"{safe_model_name}_{tensor_name}", "wb") as f:
#         pickle.dump(tensor_data, f)