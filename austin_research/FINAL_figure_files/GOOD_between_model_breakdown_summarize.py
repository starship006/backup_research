# this generates a graph to show te breakdown of self-repair across models
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %%
FOLDER_TO_STORE_PICKLES = "pickle_storage/breakdown_self_repair/"
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/comparison_graphs/"
ABLATION_TYPE = "sample"

models_to_consider = ["pythia-160m", "gpt2-small"] #  "pythia-410m"
safe_model_names = []
for model_name in models_to_consider:
    safe_model_names.append(model_name.replace("/", "_"))
# %%

tensors_to_load = {
    "condensed_logit_diff": [],
    "condensed_direct_effects": [],
    "condensed_ablated_direct_effects": [],
    "condensed_self_repair_from_heads": [],
    "condensed_self_repair_from_layers": [],
    "condensed_self_repair_from_LN": [],
    "condensed_percent_LN_of_DE": [],
    "condensed_percent_heads_of_DE": [],
    "condensed_percent_layers_of_DE": [],
}

PERCENTILE = 0.02
percentile_str = "" if PERCENTILE == 0.02 else f"{PERCENTILE}_" # 0.02 is the default

# %%
for safe_model_name in safe_model_names:
    # Loop through the dictionary and save each tensor
    for tensor_name in tensors_to_load:
        with open(FOLDER_TO_STORE_PICKLES + f"{percentile_str}{safe_model_name}_{tensor_name}", "rb") as f:
            new_tensor = pickle.load(f)
            tensors_to_load[tensor_name].append(new_tensor)
        
            
# %%
layers_per_model = []
heads_per_layer_per_model = []

for tensor in tensors_to_load["condensed_logit_diff"]:
    layers, heads_per_layer = tensor.shape
    layers_per_model.append(layers)
    heads_per_layer_per_model.append(heads_per_layer)
# %% Plot the self-repair percentage per layer per model due to LayerNorm 
percent_from_LNs = []

for i in range(len(tensors_to_load["condensed_self_repair_from_LN"])):
    self_repair = tensors_to_load["condensed_self_repair_from_LN"][i]
    direct_effect = tensors_to_load["condensed_direct_effects"][i]
    percent_from_LN = self_repair / direct_effect
    percent_from_LN = np.clip(percent_from_LN, 0, 1)
    percent_from_LNs.append(percent_from_LN)



def plot_heads_and_models(intended_value_list, title = "Graph", x_axis_title = "Layer", y_axis_title = "Something"):
    fig = go.Figure()
    colors = ["red", "blue", "green"]

    for i in range(len(tensors_to_load["condensed_self_repair_from_LN"])):
        
        data_for_model = intended_value_list[i]
        layer_of_head = einops.repeat(np.arange(layers_per_model[i]), "l -> l h", h = heads_per_layer_per_model[i])
        
        means_per_layer = data_for_model.mean(-1) # Calculate mean percent_from_LN per layer

        fig.add_trace(go.Scatter(x=layer_of_head.flatten(), y=data_for_model.flatten(), mode="markers", name=models_to_consider[i],
                                line=dict(color=colors[i], width=2), marker=dict(size=4, opacity=0.4)))
        fig.add_trace(go.Scatter(x=np.arange(layers_per_model[i]), y=means_per_layer, mode="lines", name='Mean ' + models_to_consider[i],
                                line=dict(color=colors[i], width=2)))
    fig.update_xaxes(title = x_axis_title)
    fig.update_yaxes(range=[-0.5, 1.2], title = y_axis_title)
    fig.update_layout(title = title)
    fig.show()
    

plot_heads_and_models(tensors_to_load["condensed_self_repair_from_LN"], title = "Self-Repair from LayerNorm (in Logits)", y_axis_title = "Self-Repair from LayerNorm (in Logits)")
plot_heads_and_models(percent_from_LNs, title = "Self-Repair from LayerNorm (in % of DE)", y_axis_title = "Self-Repair from LayerNorm (in % of DE)")
plot_heads_and_models(tensors_to_load["condensed_percent_LN_of_DE"], title = "CORRECT Self-Repair from LayerNorm (in % of DE)", y_axis_title = "Self-Repair from LayerNorm (in % of DE)")
# %% Now plot for MLP erasure
plot_heads_and_models(tensors_to_load["condensed_self_repair_from_layers"], title = "Self-Repair from MLP Erasure (in Logits)", y_axis_title = "Self-Repair from MLP Erasure (in Logits)")
# %%
plot_heads_and_models(tensors_to_load["condensed_percent_layers_of_DE"], title = "Self-Repair from MLP Erasure (in % of DE)", y_axis_title = "Self-Repair from MLP Erasure (in % of DE)")

    #self_repair_percentage_per_layer_per_model.append(data.mean(axis=1))
# %% Using plotly, graph the self-repair percentage per layer per model

# fig = go.Figure()
# for i in range(len(self_repair_percentage_per_layer_per_model)):
#     self_repair_percentage = self_repair_percentage_per_layer_per_model[i]
#     fig.add_trace(go.Scatter(x=np.arange(len(self_repair_percentage)), y=self_repair_percentage, mode="lines+markers", name=models_to_consider[i]))
    

# fig.update_layout(
#     title="Average Self-Repair as a fraction of the Direct Effect Per Layer Across Models",
#     xaxis_title="Layer",
#     yaxis_title="Self-Repair Percentage",
# )

# fig.show()

# %%
