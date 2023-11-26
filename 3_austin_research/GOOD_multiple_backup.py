"""
falsify/support the claim that a head can get backed up by multiple different heads
"""

# %%
from imports import *
%load_ext autoreload
%autoreload 2
from GOOD_helpers import *
# %% Constants
in_notebook_mode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "gpt-neo-125m"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name

BATCH_SIZE = 20
PROMPT_LEN = 50

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
owt_dataset = utils.get_dataset("owt")
owt_dataset_name = "owt"
# %%
all_owt_tokens = model.to_tokens(owt_dataset[0:(BATCH_SIZE * 2)]["text"]).to(device)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
# %%
logits, cache = model.run_with_cache(owt_tokens)
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=owt_tokens, model = model, display = in_notebook_mode, collect_individual_neurons = False)
# %%
ablate_heads = [[9,0]]#[i for i in top_heads if i[0] != 11]
new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in ablate_heads], return_item, corrupted_owt_tokens, apply_metric_to_cache = True)
ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling=cache)
# %%

head_input = (ablated_per_head_direct_effect - per_head_direct_effect)
layer_input = (ablated_all_layer_direct_effect - all_layer_direct_effect)

flattened_head_input = einops.rearrange(head_input, "l h a b -> l h (a b)")
flattened_layer_input = einops.rearrange(layer_input, "l a b -> l (a b)")
flattened_direct_effect = einops.rearrange(per_head_direct_effect, "l h a b -> l h (a b)")

_, top_instances = torch.topk(flattened_direct_effect[ablate_heads[0][0], ablate_heads[0][1]], 40)
filtered_head_input = flattened_head_input[:,:,top_instances]
filtered_layer_input = flattened_layer_input[:,top_instances]
# %%
def generate_interactive_heatmap(flattened_head_input, flattened_layer_input, title="Values"):
    # Define the number of steps for the slider (based on the flattened size)
    steps = flattened_head_input.size(-1)

    # Create the subplot for interactive visualization
    fig = make_subplots(rows=1, cols=1)

    # Function to update the heatmap
    def create_heatmap(step):
        head_data = flattened_head_input[..., step].cpu().numpy()
        layer_data = flattened_layer_input[..., step].cpu().numpy()

        combined_data = np.hstack([head_data, layer_data[:, np.newaxis]])

        heatmap = go.Heatmap(z=combined_data, colorscale='RdBu', zmid=0)
        return heatmap

    # Add all heatmaps to the figure but make them invisible
    for i in range(steps):
        fig.add_trace(create_heatmap(i), 1, 1)
        fig.data[i].visible = False

    # Make the first heatmap visible
    fig.data[0].visible = True

    # Define the steps for the slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"{title} - Step: {i}"}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i-th trace to "visible"
        steps.append(step)

    # Create and add slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Step: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=f"{title} - Step: 0",
        xaxis_title="Head",
        yaxis_title="Layer",
        yaxis_autorange='reversed'
    )
    
    fig.add_shape(
        go.layout.Shape(
            type="rect",
            xref="x",
            yref="y",
            x0=filtered_head_input.shape[1] + 1 - 1.5, # hack to account for mlp is to just add 1 right now
            x1=filtered_head_input.shape[1] + 1 - 0.5,
            y0=-0.5,
            y1=filtered_head_input.shape[0] - 0.5,
            line=dict(color="Black", width=2)
        )
    )

    return fig


# Usage example (assuming you have head_input and layer_input tensors ready)
fig = generate_interactive_heatmap(filtered_head_input, filtered_layer_input, f"Difference when ablating head {ablate_heads} on OWT")
# %%
fig.show()
# %% 
# save the figure as html
fig.write_html(f"GOOD_clean_figures/head_ablation_{owt_dataset_name}_{safe_model_name}.html")

# %%
