# this generates a graph to show te breakdown of self-repair across models
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %%
FOLDER_TO_STORE_PICKLES = "pickle_storage/breakdown_self_repair/"
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/comparison_graphs/"
ABLATION_TYPE = "sample"

models_to_consider = ["pythia-160m", "gpt2-small", "pythia-410m", "gpt2-medium"] #  
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
    
    "condensed_percent_self_repair_of_DE": [],
    #"full_percent_self_repair_of_DE": [],
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



def plot_heads_and_models(intended_value_list, title="Graph", x_axis_title="Layer", y_axis_title="Something", y_range=[0, 1],
                          rescaled_x=False):
    fig = go.Figure()
    colors = px.colors.qualitative.Dark2
    
    
    marker_style = dict(size=3, opacity=0.2)
    line_style = dict(width=2)
    
    
    head_traces = []
    mean_traces = []

    for i in range(len(tensors_to_load["condensed_self_repair_from_LN"])):
        data_for_model = intended_value_list[i]
        layer_of_head = einops.repeat(np.arange(layers_per_model[i]), "l -> l h", h=heads_per_layer_per_model[i])
        means_per_layer = data_for_model.mean(-1)  # Calculate mean percent_from_LN per layer

        x_layer_tensor_for_heads = layer_of_head.flatten()
        x_layer_tensor_for_mean = np.arange(layers_per_model[i])

        if rescaled_x:
            x_layer_tensor_for_heads = x_layer_tensor_for_heads / layers_per_model[i]
            x_layer_tensor_for_mean = x_layer_tensor_for_mean / layers_per_model[i]

        # Use a single trace for markers and lines
        
        head_traces.append(go.Scatter(x=x_layer_tensor_for_heads, y=data_for_model.flatten(), mode="markers",
                                 name=models_to_consider[i], line=dict(color=colors[i], **line_style),
                                 marker=dict(color=colors[i], **marker_style)))
        
        # Plot the mean line
        mean_traces.append(go.Scatter(x=x_layer_tensor_for_mean, y=means_per_layer, mode="lines",
                                 name='Mean ' + models_to_consider[i], line=dict(color=colors[i], **line_style)))

    for trace in head_traces:
        fig.add_trace(trace)
        
    for trace in mean_traces:
        fig.add_trace(trace)
        
    # Update layout for better aesthetics
    fig.update_xaxes(title=x_axis_title, showgrid = False, showline=False)
    if rescaled_x:
        fig.update_xaxes(tickformat=".0%", range = [0, 1])
    fig.update_yaxes(range=y_range, title=y_axis_title, tickformat=".0%")
    
    fig.update_yaxes(ticksuffix = "  ")
    
    
    fig.update_layout(title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    
    fig.update_layout(
        autosize=False,
        width=900,
        height=600,
        
    )

    # Show the plot
    fig.show()
    return fig
# %%   Plot for self-repair entirely
fig = plot_heads_and_models(tensors_to_load["condensed_percent_self_repair_of_DE"], title = "Self-Repair from Heads (as % of DE)",
                      y_axis_title = "Self-Repair from Head (as % of DE)", x_axis_title= "%th Layer in Model", rescaled_x=True)

fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "self_repair_full_comparison.pdf")
#plot_heads_and_models(tensors_to_load["full_percent_self_repair_of_DE"], title = "Self-Repair from Heads (in % of DE)", y_axis_title = "Self-Repair from Head")



# %%
#plot_heads_and_models(tensors_to_load["condensed_self_repair_from_LN"], title = "Self-Repair from LayerNorm (in Logits)", y_axis_title = "Self-Repair from LayerNorm (in Logits)", rescaled_x=True)
#plot_heads_and_models(percent_from_LNs, title = "Self-Repair from LayerNorm (in % of DE)", y_axis_title = "Self-Repair from LayerNorm (in % of DE)")
plot_heads_and_models(tensors_to_load["condensed_percent_LN_of_DE"], title = "Self-Repair from LayerNorm (in % of DE)", y_axis_title = "Self-Repair from LayerNorm (in % of DE)", rescaled_x=True)
# %% Now plot for MLP erasure
#plot_heads_and_models(tensors_to_load["condensed_self_repair_from_layers"], title = "Self-Repair from MLP Erasure (in Logits)", y_axis_title = "Self-Repair from MLP Erasure (in Logits)")
plot_heads_and_models(tensors_to_load["condensed_percent_layers_of_DE"], title = "Self-Repair from MLP Erasure (in % of DE)", y_axis_title = "Self-Repair from MLP Erasure (in % of DE)", rescaled_x=True)

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
