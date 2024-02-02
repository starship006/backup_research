"""
This code analyzes how the ablation of attention head leads to changes in the residual stream norm.
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, show_input, create_layered_scatter, replace_output_hook, prepare_dataset
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/"


if in_notebook_mode:
    model_name = "pythia-160m"
    BATCH_SIZE = 15
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size

# Ensure that ABLATION_TYPE is one of the expected values

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
dataset = utils.get_dataset("pile")
dataset_name = "The Pile"
all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
# %% Helper Functions

def get_norm_from_cache(cache: ActivationCache) -> Float[Tensor, "batch pos"]:
    final_resid = cache[utils.get_act_name("resid_post", model.cfg.n_layers - 1)]
    return final_resid.norm(dim = -1)


def new_cache_upon_ablation_calc(clean_tokens, corrupted_tokens, ablation_type: str, heads = None, mlp_layers = None, clean_cache: ActivationCache = None):
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]
    if ablation_type == "zero" or ablation_type == "mean":
        num_runs = 1 # since zero or mean ablation would be deterministic
    
    # Shuffle owt_tokens by batch
    shuffled_corrupted_tokens = shuffle_owt_tokens_by_batch(corrupted_tokens)
    # Calculate new_logits using act_patch
    new_cache = None
    if ablation_type == "zero":
        new_cache = act_patch(model, clean_tokens, nodes, return_item, new_cache = "zero", apply_metric_to_cache=True)
        new_logits = act_patch(model, clean_tokens, nodes, return_item, new_cache = "zero", apply_metric_to_cache=False)
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
        new_cache = act_patch(model, clean_tokens, nodes, return_item, shuffled_corrupted_tokens, apply_metric_to_cache=True)
        new_logits = act_patch(model, clean_tokens, nodes, return_item, shuffled_corrupted_tokens, apply_metric_to_cache=False)
    assert isinstance(new_cache, ActivationCache)
    return new_cache, new_logits


# %% We need to iterate through the dataset

PROMPT_LEN = 400
# %% Run. This takes a while.
ABLATION_TYPES = ["mean", "zero", "sample"]
layer = 11
head = 0

PROMPT_LEN = 100
TOTAL_TOKENS = ((10_000 // (PROMPT_LEN * BATCH_SIZE)) + 1) * (PROMPT_LEN * BATCH_SIZE)
dataset, num_batches = prepare_dataset(model, device, TOTAL_TOKENS, BATCH_SIZE, PROMPT_LEN, False, "pile")


# %%

norm_ratios = torch.zeros((num_batches, len(ABLATION_TYPES), BATCH_SIZE, PROMPT_LEN))
direct_effects = torch.zeros((num_batches, len(ABLATION_TYPES), BATCH_SIZE, PROMPT_LEN-1))
logit_diff = torch.zeros((num_batches, len(ABLATION_TYPES), BATCH_SIZE, PROMPT_LEN-1))

pbar = tqdm(total=num_batches, desc='Processing batches')
for batch_idx, clean_tokens, corrupted_tokens in dataset:
    # Cache clean/corrupted model activations + direct effects
    logits, cache = model.run_with_cache(clean_tokens)
    #corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
    
    clean_norm = get_norm_from_cache(cache)
    direct_effect_all_heads, _ = collect_direct_effect(cache, clean_tokens, model, display = False, cache_for_scaling = cache)
    clean_correct_logits = get_correct_logit_score(logits, clean_tokens)
    
    new_norms = []
    
    for ablation_type in ABLATION_TYPES:
        assert ablation_type in ["mean", "zero", "sample"], "Ablation type must be 'mean', 'zero', or 'sample'."
        new_cache, new_logit = new_cache_upon_ablation_calc(clean_tokens, corrupted_tokens, ablation_type, heads = [(layer, head)], clean_cache = cache)
        ablation_norm = get_norm_from_cache(new_cache)
        #ablated_direct_effect_head, _ = collect_direct_effect(new_cache, clean_tokens, model, display = False, cache_for_scaling = cache)
        ablated_correct_logits = get_correct_logit_score(new_logit, clean_tokens)
        
        
        
        norm_ratios[batch_idx, ABLATION_TYPES.index(ablation_type), :, :] = (ablation_norm / clean_norm).cpu()
        direct_effects[batch_idx, ABLATION_TYPES.index(ablation_type), :, :] = (direct_effect_all_heads[layer, head]).cpu()
        logit_diff[batch_idx, ABLATION_TYPES.index(ablation_type), :, :] = (ablated_correct_logits - clean_correct_logits).cpu()
        
    pbar.update(1)
        
    
# make histogram for LN ablation comparison
new_norms = [norm_ratios[:, i, ...].flatten() for i in range(len(ABLATION_TYPES))]
hist_data = [go.Histogram(x=norm, name=ablation_type, opacity=0.7) for ablation_type, norm in zip(ABLATION_TYPES, new_norms)]

layout = go.Layout(
    title=f'Norm of Final Residual Stream when Ablation Layer {layer}, Head {head}',
    xaxis=dict(title='Ratio of Ablated to Clean Norm'),
    yaxis=dict(title='Frequency'),
    barmode='overlay'  # Overlay histograms for better comparison
)

fig = go.Figure(data=hist_data, layout=layout)
fig.show()


# make histogram for direct effect vs logit diff comparison
        
# %%
PROPER_ABLATION_TITLES = ['mean', 'zero', 'resample']
fig = plotly.subplots.make_subplots(
    rows=1, cols=3,
    shared_xaxes=False,
    shared_yaxes=True,
    vertical_spacing=0.25,
    horizontal_spacing=0.05,
    #specs=[[{"type": "scatter"}] * 2]*2,
    subplot_titles=PROPER_ABLATION_TITLES,
)
colors = ['purple', 'light blue', 'pink', ]

for i in range(3):
    
    logit_diffs = logit_diff[:, i, :, :].flatten()
    direct_effect = direct_effects[:, i, :, :].flatten()
    ablation = PROPER_ABLATION_TITLES[i]

    fig.add_trace(go.Scatter(
        x=direct_effect,
        y=logit_diffs,
        mode='markers',
        name=ablation,
        marker=dict(size=2, color=colors[i]),
    ), row=1, col = i+1)
    
    
for i in range(3):
    # Line plot for y=-x
    fig.add_trace(go.Scatter(
        x=[-4, 4],  # Adjust the range as needed
        y=[4, -4],
        mode='lines',
        line=dict(color='black', width=1, dash='dash'),
        name='y=-x',
        
    ), row=1, col=i+1)
    

    

fig.update_layout(
        title=f"Self-Repair when ablating L{layer}H{head} in {model_name}",
        xaxis_title="Direct Effect",
        yaxis_title="Logit Difference",
        # Additional layout parameters for better visualization
        barmode='overlay',
        width = 700,
        height = 450,
        plot_bgcolor='white',  # Add this line to set the background color to white
    )
fig.update_xaxes(title_text="Direct Effect", zeroline=True, zerolinecolor='black', range=[-4, 4], linecolor='black')
fig.update_yaxes(title_text="Logit Difference", zeroline = True, zerolinecolor='black', range=[-4, 4], linecolor='black')

fig.show()
fig.write_image(f"{FOLDER_TO_WRITE_GRAPHS_TO}self_repair_ablation_L{layer}H{head}_{model_name}.pdf")

# %%