# this generates a graph to show te breakdown of self-repair across models
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %%
FOLDER_TO_STORE_PICKLES = "pickle_storage/breakdown_self_repair/"
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/comparison_graphs/"
ABLATION_TYPE = "sample"

models_to_consider = ["pythia-160m", "gpt2-small", "pythia-410m", "gpt2-medium", "gpt2-large", "pythia-1b", "llama-7b"] #  
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
    
    "full_logit_diff": [],
    "full_direct_effects": [],
    "full_ablated_direct_effects": [],
    "full_self_repair_from_heads": [],
    "full_self_repair_from_layers": [],
    "full_self_repair_from_LN": [],
    
    "full_percent_LN_of_DE": [],
    "full_percent_heads_of_DE": [],
    "full_percent_layers_of_DE": [],
    "full_percent_self_repair_of_DE": [],
    
    "full_unclipped_percent_LN_of_DE": [],
    "full_unclipped_percent_heads_of_DE": [],
    "full_unclipped_percent_layers_of_DE": [],
    "full_unclipped_percent_self_repair_of_DE": [],
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

# %%

def plot_heads_and_models(intended_value_list, title="Graph", x_axis_title="Layer", y_axis_title="Something", y_range=[0, 1],
                          rescaled_x=False, y_percentage = True, default_scaling = False):
    fig = go.Figure()
    colors = px.colors.qualitative.Dark2

    marker_style = dict(size=2, opacity=1)
    line_style = dict(width=3)

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
                                 name=models_to_consider[i], legendgroup=i,
                                 line=dict(color=colors[i], **line_style),
                                 marker=dict(color=colors[i], **marker_style)))
        
        # Plot the mean line
        mean_traces.append(go.Scatter(x=x_layer_tensor_for_mean, y=means_per_layer, mode="lines",
                                 name=models_to_consider[i], legendgroup=i,
                                 line=dict(color=colors[i], **line_style)))

    for trace in head_traces:
        fig.add_trace(trace)
        
    for trace in mean_traces:
        fig.add_trace(trace)
        
    # Update layout for better aesthetics
    fig.update_xaxes(title=x_axis_title, showgrid = False, zeroline = False, showline=True, linecolor = 'black')
    if rescaled_x:
        fig.update_xaxes(tickformat=".0%", range = [-0.02, 1], tickfont=dict(size=18))
    fig.update_yaxes(title=y_axis_title, tickformat=".0%" if y_percentage else "", tickfont=dict(size=18),
                     zerolinecolor = 'Grey', linecolor = 'black')
    fig.update_layout(title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),)
    
    
    #fig.update_xaxes(showgrid=True, gridcolor='black')
    #fig.update_yaxes(showgrid=True, gridcolor='black')


    if not default_scaling:
        fig.update_yaxes(range=y_range,)
    
    
    
    
    fig.update_layout(
        autosize=False,
        width=1300,
        height=700,
        font=dict(
            #family="Courier New, monospace",
            size=18,
            #color="RebeccaPurple"
        ),
        plot_bgcolor='white',  # Add this line to set the background color to white

    )

    # Show the plot
    fig.show()
    return fig


# %% Plot for self-repair entirely
fig = plot_heads_and_models(tensors_to_load["condensed_percent_self_repair_of_DE"], title = "Self-Repair (as % of DE)",
                      y_axis_title = "Self-Repair (as % of DE)", x_axis_title= "%th Layer in Model", rescaled_x=True)
# %%
fig = plot_heads_and_models(tensors_to_load["full_percent_self_repair_of_DE"], title = "Self-Repair (as % of DE)",
                      y_axis_title = "Self-Repair (as % of DE)", x_axis_title= "%th Layer in Model", rescaled_x=True)
# %%
fig = plot_heads_and_models(tensors_to_load["full_unclipped_percent_self_repair_of_DE"], title = "",
                      y_axis_title = "Self-Repair (as % of DE)", x_axis_title= "%th Layer in Model", rescaled_x=True, default_scaling=True)

# %% Try to put a number behind the self-repairing of direct effects
for model_index in range(len(tensors_to_load["full_unclipped_percent_self_repair_of_DE"])):
    self_repair_model = tensors_to_load["full_unclipped_percent_self_repair_of_DE"][model_index]
    print(self_repair_model)
    print(f"Model {models_to_consider[model_index]} has a mean self-repair of {self_repair_model.mean():.2%} and a median self-repair of {np.median(self_repair_model):.2%}")


# %% For the full self-repair just in terms of logits
condensed_self_repair_as_logits = []
full_self_repair_as_logits = []

for i in range(len(tensors_to_load["condensed_self_repair_from_LN"])):
    condensed_self_repair = tensors_to_load["condensed_logit_diff"][i] - (tensors_to_load["condensed_ablated_direct_effects"][i] - tensors_to_load["condensed_direct_effects"][i])
    full_self_repair = tensors_to_load["full_logit_diff"][i] - (tensors_to_load["full_ablated_direct_effects"][i] - tensors_to_load["full_direct_effects"][i])
    
    condensed_self_repair_as_logits.append(condensed_self_repair)
    full_self_repair_as_logits.append(full_self_repair)




#fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "self_repair_full_comparison.pdf")
#plot_heads_and_models(tensors_to_load["full_percent_self_repair_of_DE"], title = "Self-Repair from Heads (in % of DE)", y_axis_title = "Self-Repair from Head")
# %% with logits on FULL
fig = plot_heads_and_models(full_self_repair_as_logits, title = "Self-Repair (as logits)",
                      y_axis_title = "Self-Repair (as logits)", x_axis_title= "%th Layer in Model", rescaled_x=True, y_percentage=False, y_range=[-1, 1])

#fig.add_annotation(x=.2, y=0.2, text="Early-Layer Breakage", font=dict(color="red"), xref="paper", yref="paper", showarrow=False)
#fig.add_annotation(x=.8, y=.95, text="Self-Repairing", font=dict(color="#007BFF"), xref="paper", yref="paper", showarrow = False)#fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "self_repair_full_comparison_logits.pdf")



# %% with logits, but condensed
fig = plot_heads_and_models(condensed_self_repair_as_logits, title = "",
                      y_axis_title = "Self-Repair (as logits)", x_axis_title= "%th Layer in Model", rescaled_x=True, y_percentage=False, y_range=[-1, 1.75])

fig.add_annotation(x=.2, y=0.2, text="Early-Layer Breakage", font=dict(color="red", size = 20), xref="paper", yref="paper", showarrow=False)
fig.add_annotation(x=.8, y=.8, text="Self-Repairing Behavior", font=dict(color="#007BFF", size = 20), xref="paper", yref="paper", showarrow = False)#fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "self_repair_full_comparison_logits.pdf")

fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "condensed_self_repair_as_logits_GRAPH.pdf")



# %% Next, lets make a graph which compares the values of the self-repair from heads/mlp/LN to the total self-repair
fractions_from_heads = []
fractions_from_layers = []
fractions_from_LN = []

mean_in_second_to_last_layer_head = []
mean_in_second_to_last_layer_layers = []
mean_in_second_to_last_layer_LN = []

mean_in_second_to_last_layer_head_wrt_DE = []
mean_in_second_to_last_layer_layers_wrt_DE = []
mean_in_second_to_last_layer_LN_wrt_DE = []


for model_index in range(len(tensors_to_load["full_unclipped_percent_self_repair_of_DE"])):
    self_repair_from_heads = tensors_to_load["condensed_self_repair_from_heads"][model_index]
    self_repair_from_layers = tensors_to_load["condensed_self_repair_from_layers"][model_index]
    self_repair_from_LN = tensors_to_load["condensed_self_repair_from_LN"][model_index]
    self_repair_total = self_repair_from_heads + self_repair_from_layers + self_repair_from_LN
    
    #fraction_from_heads = np.clip((self_repair_from_heads / self_repair_total), 0, 1)
    #fraction_from_layers = np.clip(self_repair_from_layers / self_repair_total, 0, 1)
    #fraction_from_LN = np.clip(self_repair_from_LN / self_repair_total, 0, 1)
    
    fraction_from_heads = self_repair_from_heads / self_repair_total
    fraction_from_layers = self_repair_from_layers / self_repair_total
    fraction_from_LN = self_repair_from_LN / self_repair_total
    
    fractions_from_heads.append(fraction_from_heads)
    fractions_from_layers.append(fraction_from_layers)
    fractions_from_LN.append(fraction_from_LN)
    
    mean_in_second_to_last_layer_head.append(fraction_from_heads.mean(-1)[-2])
    mean_in_second_to_last_layer_layers.append(fraction_from_layers.mean(-1)[-2])
    mean_in_second_to_last_layer_LN.append(fraction_from_LN.mean(-1)[-2]) 
    
    # look at stuff w.r.t DE
    direct_effect = tensors_to_load["condensed_direct_effects"][model_index]
    mean_in_second_to_last_layer_head_wrt_DE.append((self_repair_from_heads / direct_effect).mean(-1)[-2])
    mean_in_second_to_last_layer_layers_wrt_DE.append((self_repair_from_layers / direct_effect).mean(-1)[-2])
    mean_in_second_to_last_layer_LN_wrt_DE.append((self_repair_from_LN / direct_effect).mean(-1)[-2])
    
    
# %%
print("LN explains " + str(torch.stack(mean_in_second_to_last_layer_LN_wrt_DE).mean().item()) + " percent of DE in the second-to-last layer on average in models")
    
# %%
"""

NOTE: A LOT OF THE BOTTOM ISN'T USED. this is messy. i may include them in appendix or something.

"""

# %%
def plot_multiple_things(list_of_intended_value_list,  title="Graph", x_axis_title="Layer", y_axis_title="Something", y_range=[0, 1],
                          rescaled_x=False, y_percentage = True, default_scaling = False):
    fig = go.Figure()
    colors = px.colors.qualitative.Dark2
    marker_style = dict(size=3, opacity=0.25)
    line_style = dict(width=3)
    dashes = ['dash', 'dot', 'dashdot']
    
    
    for index_trace, intended_value_list in enumerate(list_of_intended_value_list):
        head_traces = []
        mean_traces = []


        for i in range(len(intended_value_list)):
            data_for_model = intended_value_list[i]
            layer_of_head = einops.repeat(np.arange(layers_per_model[i]), "l -> l h", h=heads_per_layer_per_model[i])
            means_per_layer = data_for_model.mean(-1)  # Calculate mean percent_from_LN per layer

            x_layer_tensor_for_heads = layer_of_head.flatten()
            x_layer_tensor_for_mean = np.arange(layers_per_model[i])

            if rescaled_x:
                x_layer_tensor_for_heads = x_layer_tensor_for_heads / layers_per_model[i]
                x_layer_tensor_for_mean = x_layer_tensor_for_mean / layers_per_model[i]

            # Use a single trace for markers and lines
            # head_traces.append(go.Scatter(x=x_layer_tensor_for_heads, y=data_for_model.flatten(), mode="markers",
            #                         name=models_to_consider[i], legendgroup=i,
            #                         line=dict(color=colors[i], **line_style),
            #                         marker=dict(color=colors[i], **marker_style)))
            
            # Plot the mean line
            mean_traces.append(go.Scatter(x=x_layer_tensor_for_mean, y=means_per_layer, mode="lines",
                                    name=models_to_consider[i], legendgroup=i,
                                    line=dict(color=colors[i], dash = dashes[index_trace], **line_style)))

        # for trace in head_traces:
        #     fig.add_trace(trace)
            
        for trace in mean_traces:
            fig.add_trace(trace)
        
    # Update layout for better aesthetics
    fig.update_xaxes(title=x_axis_title, showgrid = False, zeroline = False, showline=False, linecolor = 'black')
    if rescaled_x:
        fig.update_xaxes(tickformat=".0%", range = [-0.02, 1], tickfont=dict(size=18))
    fig.update_yaxes(title=y_axis_title, tickformat=".0%" if y_percentage else "", tickfont=dict(size=18),
                     zerolinecolor = 'Grey', linecolor = 'black')
    fig.update_layout(title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),)
    
    if not default_scaling:
        fig.update_yaxes(range=y_range,)
    
    
    fig.update_layout(
        autosize=False,
        width=1300,
        height=700,
        font=dict(
            #family="Courier New, monospace",
            size=18,
            #color="RebeccaPurple"
        ),
        plot_bgcolor='white',  # Add this line to set the background color to white

    )

    # Show the plot
    fig.show()
    return fig


# %%
plot_multiple_things([fractions_from_heads, fractions_from_layers, fractions_from_LN], title = "Self-Repair Breakdown",
                     rescaled_x = True)



# %%





# %% get at a number for how important LN is.
for model_index in range(len(tensors_to_load["full_unclipped_percent_self_repair_of_DE"])):
    self_repair_model = tensors_to_load["full_unclipped_percent_self_repair_of_DE"][model_index]
    print(self_repair_model)
    print(f"Model {models_to_consider[model_index]} has a mean self-repair of {self_repair_model.mean():.2%} and a median self-repair of {np.median(self_repair_model):.2%}")






# %% In Logits, Condensed
fig = plot_heads_and_models(tensors_to_load["condensed_self_repair_from_LN"], title = "", y_axis_title = "Self-Repair from LayerNorm (in Logits)", rescaled_x=True, y_percentage=False, y_range=[-1,1])
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "condensed_self_repair_from_LN_graph.pdf")

# %% In Logits, Full
fig = plot_heads_and_models(tensors_to_load["full_self_repair_from_LN"], title = "Self-Repair from LayerNorm (in Logits)", y_axis_title = "Self-Repair from LayerNorm (in Logits)", rescaled_x=True, y_percentage=False, y_range=[-1,1])
# %% In Percent, Condensed

fig = plot_heads_and_models(tensors_to_load["condensed_percent_LN_of_DE"], title = "", y_axis_title = "Self-Repair from LayerNorm (in % of DE)", rescaled_x=True)
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "condensed_percent_LN_of_DE_GRAPH.pdf")
# %% In Percent, Full
fig = plot_heads_and_models(tensors_to_load["full_percent_LN_of_DE"], title = "Self-Repair from LayerNorm (in % of DE)", y_axis_title = "Self-Repair from LayerNorm (in % of DE)", rescaled_x=True)
# %% In Percent, Full, Unclipped
fig = plot_heads_and_models(tensors_to_load["full_unclipped_percent_LN_of_DE"], title = "", y_axis_title = "Self-Repair from LayerNorm (in % of DE)", rescaled_x=True, y_range = [-150, 150])
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "full_unclipped_percent_LN_of_DE.pdf")


# %% Now plot for MLP erasure
# %% In Percent, Condensed
fig = plot_heads_and_models(tensors_to_load["condensed_percent_layers_of_DE"], title = "", y_axis_title = "Self-Repair from MLP Erasure (in % of DE)", rescaled_x=True)
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "condensed_percent_layers_of_DE_GRAPH.pdf")
# %% In Logits, Condensed
fig = plot_heads_and_models(tensors_to_load["condensed_self_repair_from_layers"], title = "", y_axis_title = "Self-Repair from MLP Erasure (in Logits)", rescaled_x=True, y_percentage=False, y_range=[-1,1])
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + "condensed_self_repair_from_layers_GRAPH.pdf")

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
