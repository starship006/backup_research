"""
This code breaks processes the tensors from GOOD_MLP_erasure_breakdown.py and generates the graphs for the paper. It turns out
the original tensors were insane and I need to figure out why
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, show_input, collect_direct_effect, get_single_correct_logit, topk_of_Nd_tensor, return_item, get_correct_logit_score, prepare_dataset
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/mlp_sparsity/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/mlp_sparsity/"


if in_notebook_mode:
    model_name = "pythia-160m"
    BATCH_SIZE = 20
    #PERCENTILE = 0.01
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)  
    parser.add_argument('--percentile', type=float, default=0.02)
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    #PERCENTILE = args.percentile
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
prompt_len = PROMPT_LEN = 2 #random value thatll get changed
num_batches = 1 #random value thatll get changed
PADDING = False
num_prompts = 1

# correct values
ablate_layer = model.cfg.n_layers - 2 # second to last layer
num_heads = model.cfg.n_heads
# %% Load tensors from storage
logit_diffs_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)
direct_effects_across_everything = torch.zeros(model.cfg.n_heads,num_prompts, PROMPT_LEN - 1)
ablated_direct_effects_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)
self_repair_from_heads_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)
self_repair_from_layers_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)
direct_effects_from_layers_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)
self_repair_from_LN_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)
self_repair_across_everything = torch.zeros(model.cfg.n_heads, num_prompts, PROMPT_LEN - 1)

N = 20  # Example value

# Initialize the result tensors with the appropriate sizes

top_neurons_idx = torch.zeros((num_heads, num_prompts, prompt_len, N), dtype=torch.int64)
top_neuron_vals = torch.zeros((num_heads, num_prompts, prompt_len, N))
top_neuron_initial_vals = torch.zeros((num_heads, num_prompts, prompt_len, N))


tensors_to_load = {
    "logit_diffs_across_everything",
    "direct_effects_across_everything",
    "ablated_direct_effects_across_everything",
    "self_repair_from_heads_across_everything",
    "self_repair_from_layers_across_everything",
    "self_repair_from_LN_across_everything",
    "self_repair_across_everything",
    "top_neurons_idx",
    "top_neuron_vals", #val = self repair value
    "top_neuron_initial_vals",
    "direct_effects_from_layers_across_everything"
}
# %%
FOLDER_TO_STORE_PICKLES = "pickle_storage/mlp_sparsity/"
subfolder_path = Path(FOLDER_TO_STORE_PICKLES) / safe_model_name
subfolder_path.mkdir(parents=True, exist_ok=True)  # This creates the subfolder if it doesn't exist

for tensor_name in tensors_to_load:
    file_path = subfolder_path / f"MLPs_{tensor_name}_{safe_model_name}_L{ablate_layer}.pickle"
    with open(file_path, "rb") as f:
        locals()[tensor_name] = pickle.load(f)    
# %% 
num_heads, num_prompts, prompt_len_minus_one = logit_diffs_across_everything.shape
prompt_len = prompt_len_minus_one + 1
assert num_heads == model.cfg.n_heads
# %% First: MLP explains significant portion of self-repair
num_cols = 4
num_rows = model.cfg.n_heads // num_cols

ADD_BOUNDS = True # if we should bound the values because noise
for PERCENTILE in [0.02]: # 0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
    num_prompts_to_consider = int(num_prompts * prompt_len_minus_one * PERCENTILE)
    print("Considering top", num_prompts_to_consider, "prompts")
    # Create the plot
    labels = [f"L{ablate_layer}H{h}" for h in range(model.cfg.n_heads)]
    fig = plotly.subplots.make_subplots(rows=num_rows, cols=num_cols, shared_xaxes=True, shared_yaxes=True, subplot_titles=labels)
    
    for head in range(model.cfg.n_heads):
        row = head // num_cols + 1
        col = head % num_cols + 1
        layer_self_repair = self_repair_from_layers_across_everything[head]
        total_self_repair = self_repair_across_everything[head]
        
        top_DE_in_head = topk_of_Nd_tensor(direct_effects_across_everything[head], num_prompts_to_consider)
        filtered_batch_pos = top_DE_in_head
        
        # AVERAGE SELF-REPAIR
        all_cumulative_percentages = []
        for batch, pos in filtered_batch_pos:
            summed_neuron_self_repair_instance = max(layer_self_repair[batch, pos], 0) # if negative, make zero
            self_repair_instance = total_self_repair[batch, pos]
            if self_repair_instance <= 0:
                continue
            
            percentage_from_neurons = (100. * summed_neuron_self_repair_instance / self_repair_instance).item()
            
            if ADD_BOUNDS:
                percentage_from_neurons = min(percentage_from_neurons, 100)
                percentage_from_neurons = max(percentage_from_neurons, 0)
            
            all_cumulative_percentages.append(percentage_from_neurons)
            
        all_cumulative_percentages = np.array(all_cumulative_percentages)
        average_cumulative_percentages = np.mean(all_cumulative_percentages, axis=0)
        
        print(f"Head {head} | len {len(all_cumulative_percentages)} : " + str(average_cumulative_percentages))
        fig.add_trace(go.Histogram(x=all_cumulative_percentages, name=f"L{ablate_layer}H{head}"), row=row, col=col)
        
    fig.update_layout(title=f"Percent Self-Repair Explained by MLP{model.cfg.n_layers - 1} when ablating Layer {ablate_layer} Heads in {model_name} | Top {PERCENTILE * 100}% DE",
                      xaxis_title="Percentage from Neurons",
                      yaxis_title="Count",
                      width = 800,
                      height = 600,
                      font=dict(size=10))
    fig.update_yaxes(title_text="Count", title_standoff=10, range=[0, 13400]) # set the same y-axis range for all subplots
    fig.update_xaxes(title_text="MLP Percentage", title_standoff=10)
    
    fig.show()
        
        # # Determine X values (number of indices)
        # x_values = np.arange(1, len(neuron_self_repair[:, batch, pos]) + 1)

    
        # fig.add_trace(go.Scatter(x=x_values, y=average_cumulative_percentages,
        #                     mode='lines', name='L' + str(ablate_layer) + 'H' + str(head), text = [str(i) for i in filtered_batch_pos]))
    
    # fig.update_layout(
    #     title=f"Average Cumulative Percentage of L{model.cfg.n_layers - 1} Self-Repair explained by Top-X Neurons on Top {PERCENTILE * 100}% Examples of L{ablate_layer}",
    #     xaxis_title="Top X Neurons",
    #     yaxis_title="Average Percentage of Total Sum",
    #     yaxis=dict(tickformat=".2f"),
    # )

    # if in_notebook_mode:
    #     fig.show()

    fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"boolean_activation/{safe_model_name}_L{ablate_layer}_{PERCENTILE}.html")
    fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"boolean_activation/{safe_model_name}_L{ablate_layer}_{PERCENTILE}.pdf")

# %% Next: A few neurons is often enough to explain the self-repair on a prompt

x_neurons = 5 # how many neurons to filter for
traces = [] # initialize an empty list to store the histogram traces
cap_amount = 300 # top percentile to bound at

for PERCENTILE in [0.02]: # 0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
    num_prompts_to_consider = int(num_prompts * prompt_len_minus_one * PERCENTILE)
    print("Considering top", num_prompts_to_consider, "prompts")

    for head in range(model.cfg.n_heads):
        top_neurons_self_repair = top_neuron_vals[head]
        filtered_batch_pos = topk_of_Nd_tensor(self_repair_from_layers_across_everything[head], num_prompts_to_consider)

        # AVERAGE SELF-REPAIR EXPLAINED BY TOP X NEURONS
        all_cumulative_percentages = []
        for batch, pos in filtered_batch_pos:
            total_neuron_self_repair = top_neurons_self_repair[batch, pos, :x_neurons].sum()
            percent_explained = 100 * total_neuron_self_repair / self_repair_from_layers_across_everything[head, batch, pos]
            
            if ADD_BOUNDS:
                percent_explained = min(percent_explained, cap_amount)
            all_cumulative_percentages.append(percent_explained)
            

        
        all_cumulative_percentages = np.array(all_cumulative_percentages)
        print("Average for head " + str(head) + ": " + str(np.mean(all_cumulative_percentages)))
        # Create a histogram trace for this head
        trace = go.Histogram(
            x=all_cumulative_percentages,
            name='L' + str(ablate_layer) + 'H' + str(head),
            opacity=0.75,
        )

        # Add the trace to the list
        traces.append(trace)

    # Create the figure object
    fig = go.Figure()

    # Add each trace to the figure
    for trace in traces:
        fig.add_trace(trace)

    # Set the layout of the figure
    fig.update_layout(
        title='Percentage of Self-Repair Explained by Top ' + str(x_neurons) + ' Neurons in Layer ' + str(model.cfg.n_layers - 1) + ' | Top ' + str(PERCENTILE * 100) + '%',
        xaxis_title='Percentage Explained',
        yaxis_title='Counts',
        width = 800,
        height = 600,
        barmode = 'overlay'
    )

    fig.update_xaxes(range=[0, cap_amount])
    # Display the figure
    fig.show()


    fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"sparse_neuron_activation/{safe_model_name}_L{ablate_layer}_{PERCENTILE}.pdf")    
            
    
# %% These neurons are usually anti-erasure neurons. Plotting the top self-repair neurons pre and post ablation
traces = [] # initialize an empty list to store the histogram traces


for PERCENTILE in [0.02]:  # 0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
    num_prompts_to_consider = int(num_prompts * prompt_len_minus_one * PERCENTILE)
    print("Considering top", num_prompts_to_consider, "prompts")

    for head in [11]:  # range(model.cfg.n_heads):
        clean_direct_effect = torch.zeros(num_prompts_to_consider)  # DE of top self-repair neuron on clean
        corrupted_direct_effect = torch.zeros(num_prompts_to_consider)  # DE of top self-repair neuron on corrupted
        layer_direct_effect = torch.zeros(num_prompts_to_consider)  # DE of entire layer on corrupted
        top_neurons_self_repair = top_neuron_vals[head]
        filtered_batch_pos = topk_of_Nd_tensor(self_repair_from_layers_across_everything[head], num_prompts_to_consider)

        # AVERAGE SELF-REPAIR EXPLAINED BY TOP X NEURONS
        all_cumulative_percentages = []
        for i, batch_pos in enumerate(filtered_batch_pos):
            batch, pos = batch_pos
            clean_direct_effect[i] = top_neuron_initial_vals[head, batch, pos, 0]
            corrupted_direct_effect[i] = top_neurons_self_repair[batch, pos, 0] + top_neuron_initial_vals[head, batch, pos, 0]  # self_repair = corrupted - clean
            layer_direct_effect[i] = direct_effects_from_layers_across_everything[-1, batch, pos]

        fig = px.scatter(x=clean_direct_effect, y=corrupted_direct_effect, color=layer_direct_effect, marginal_x="histogram",)#, title = "L" + str(ablate_layer) + "H" + str(head))

        fig.add_trace(go.Scatter(x=[min(clean_direct_effect), max(clean_direct_effect)], y=[min(clean_direct_effect), max(clean_direct_effect)], 
                                 mode='lines', name='y=x', line=dict(color='black', dash='dash')))
        
        fig.data[0].marker.size = 2
        # fig.update_traces(marker_size=2)
        fig.update_layout(
            # title=f'Clean/Ablated Direct Effects of top Self-Repairing Neuron when ablating L{ablate_layer}H{head} in {safe_model_name}'+ ' | Top ' + str(PERCENTILE * 100) + '%',
            xaxis_title='Clean Direct Effect',
            yaxis_title='Ablated Direct Effect',
            coloraxis=dict(
                colorscale=[
                    [0, 'rosybrown'],   # Red for negative values
                    [0.5, 'white'],
                    [0.75, 'lightblue'],  # Blue for positive values
                    [1, 'darkblue']
                ],
                cmid=0,                # Midpoint (white)
                cmin=-16,              # Minimum color (-12)
                cmax=16               # Maximum color (12)
            ),
        )
        fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=-0.2,
                                          ticks="outside", title="Last MLP DE"))

        range_size = 4.5
        fig.update_layout(
            xaxis=dict(range=[-range_size, range_size]),
            yaxis=dict(range=[-range_size, range_size]),
            plot_bgcolor='white',
            showlegend=False

        )

        fig.update_layout(
            width=1000,
            height=500,
        )

        fig.update_xaxes(linecolor='Black', zeroline=True, zerolinecolor='black',)
        fig.update_yaxes(linecolor='Black', zeroline=True, zerolinecolor='black',)

        # Add a box around the histogram
        # box_x_min, box_x_max = min(clean_direct_effect), max(clean_direct_effect)+0.3*abs(max(clean_direct_effect))
        # box_y_min, box_y_max = min(corrupted_direct_effect)-0.3*abs(min(corrupted_direct_effect)), max(corrupted_direct_effect)+0.3*abs(max(corrupted_direct_effect))

        rect = go.Layout({
            'shapes': [
                
            #     {
            #     'type': 'rect',
            #     'xref': 'x',
            #     'yref': 'y',
            #     'x0': box_x_min,
            #     'y0': 0,
            #     'x1': box_x_max,
            #     'y1': 0.1,
            #     'fillcolor': 'WhiteSmoke',
            #     'opacity': 1,
            #     'line': {'width': 2, 'color': 'Black'}
            # },
                       
            dict(type='rect',
             xref='paper', yref='paper',
             x0=0, x1 = 1, y0=0.743, y1=1,  # Adjust the height (y1 value) if needed
             line=dict(width=2, color='black'),
             fillcolor='rgba(0, 0, 0, 0)',
             opacity=1,
            ),
            ]
        })

        fig.update_layout(rect)

        fig.show()

        
# %%
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"erasure_neuron/{safe_model_name}_L{ablate_layer}H{11}_{PERCENTILE}.pdf")    


# %% It's not the case that the same neurons are performing self-repair
top_x_neurons = 10
for PERCENTILE in [0.02]: # 0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
    num_prompts_to_consider = int(num_prompts * prompt_len_minus_one * PERCENTILE)
    print("Considering top", num_prompts_to_consider, "prompts")
    print("BRUH YOU SHOULD CHANGE TITLE")
    for head in [11]:#range(model.cfg.n_heads):
        clean_direct_effect = torch.zeros(num_prompts_to_consider) # DE of top self-repair neuron on clean
        corrupted_direct_effect = torch.zeros(num_prompts_to_consider) # DE of top self-repair neuron on corrupted
        
        top_neurons_self_repair = top_neuron_vals[head]
        filtered_batch_pos = topk_of_Nd_tensor(self_repair_from_layers_across_everything[head], num_prompts_to_consider)

        all_counts = [0 for _ in range(model.cfg.d_mlp)]
        for batch, pos in filtered_batch_pos:
            for neuron in top_neurons_idx[head, batch, pos, 0:top_x_neurons]:
                all_counts[int(neuron.item())] += 1

        # Sort neurons by count
        sorted_neurons = sorted(range(len(all_counts)), key=all_counts.__getitem__, reverse=True)
        sorted_neurons_with_percents = [(sorted_neurons[i], all_counts[sorted_neurons[i]] / num_prompts_to_consider * 100) for i in range(len(sorted_neurons))]

        
        percents_to_plot = [neuron_tuple[1] for neuron_tuple in sorted_neurons_with_percents]

        fig = px.bar(x = list(range(len(sorted_neurons))), y = percents_to_plot)#,  title = "Neuron Self-Repair when ablating L" + str(ablate_layer) + "H" + str(head) + f" in {safe_model_name}")
        fig.update_traces(hovertext=[str(neuron) for neuron in sorted_neurons], hoverinfo='text')
        fig.update_yaxes(title = "Percentage of Prompts with Neuron in Top " + str(top_x_neurons))
        
        fig.update_traces(hovertext=[str(neuron) for neuron in sorted_neurons], hoverinfo='text')
        
        
        # dont show x axis label
        fig.update_xaxes(showticklabels=False, title = "Different Neurons")
        fig.show()

# %% LOAD THE SPARSITY STUFF



# %% LOAD THESE TENSORS:

tensors_to_load = {
    "llama-7b/MLPs_is_larger_tensors_llama-7b_L30_H8.pickle",
    "gpt2-small/MLPs_is_larger_tensors_gpt2-small_L9_H11.pickle",
    "pythia-160m/MLPs_is_larger_tensors_pythia-160m_L7_H8.pickle",
    "pythia-410m/MLPs_is_larger_tensors_pythia-410m_L17_H4.pickle",
}

tensor_names = {
    "llama_is_larger" ,
    "gpt2-small_is_larger",
    "pythia-160m_is_larger",
    "pythia-410m_is_larger",
}

all_is_larger_tensors = []
# %%
FOLDER_TO_STORE_PICKLES = "pickle_storage/mlp_sparsity/"
subfolder_path = Path(FOLDER_TO_STORE_PICKLES)

for tensor_name in tensors_to_load:
    file_path = subfolder_path / tensor_name
    with open(file_path, "rb") as f:
        a = pickle.load(f)   
        all_is_larger_tensors.append(a)
        
# %%
model_names = ["Llama-7b L30H8", "GPT-2 Small L9H11", "Pythia-160m L7H8", "Pythia-410m L17H4"]
percentages = [0.5, 0.1, 0.05, 0.025, 0.01]
fig = make_subplots(rows = 2, cols = 2,  vertical_spacing=0.125, horizontal_spacing=0.1, subplot_titles=model_names)
colors = ["#EDBFC6", "#66D9D7", "#678FBF", "#465DA3", "#192A5E"]


for model_index, is_larger_tensors in enumerate(all_is_larger_tensors):
    
    for i, is_larger_tensor in enumerate(is_larger_tensors):
        fig.add_trace(go.Scatter(x=np.arange(1, is_larger_tensor.shape[0]),
                             y=is_larger_tensor,
                             mode='lines+markers',
                             name=f"{percentages[i] * 100}%",
                             marker=dict(color=colors[i])
                        ), row = model_index // 2 + 1, col = model_index % 2 + 1)


    fig.update_layout(
        height=800,
        width=1000,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white',
    )
    fig.update_xaxes(range=[0, 50], title_text = "Top Nth Neuron", linecolor="Black")
    fig.update_yaxes(title_text = "% of instances", tickformat=".0%", linecolor="Black", gridcolor="grey")

if in_notebook_mode:
    fig.show()

# %%
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"top_neuron_explains_instances_combined.pdf")

# %%
