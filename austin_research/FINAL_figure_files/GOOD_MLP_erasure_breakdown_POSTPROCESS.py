"""
This code breaks processes the tensors from GOOD_MLP_erasure_breakdown.py and generates the graphs for the paper. It turns out
the original tensors were insane and I need to figure out why
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, show_input, collect_direct_effect, get_single_correct_logit, topk_of_Nd_tensor, return_item, get_correct_logit_score
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

# %% We need to iterate through the dataset to find the 
TOTAL_PROMPTS_TO_ITERATE_THROUGH = 3000
PROMPT_LEN = 400

num_batches = TOTAL_PROMPTS_TO_ITERATE_THROUGH // BATCH_SIZE
ablate_layer = model.cfg.n_layers - 2
# %% Load tensors from storage

tensors_to_load = ["last_layer_clean_neurons_across_everything", "last_layer_ablated_neurons_across_everything", "self_repair_across_everything", "direct_effects_across_everything", "self_repair_from_layers_across_everything"]

direct_effects_across_everything = torch.zeros(model.cfg.n_heads,TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1).to(device)
self_repair_from_layers_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
self_repair_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
last_layer_clean_neurons_across_everything = torch.zeros(model.cfg.n_heads, model.cfg.d_mlp, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
last_layer_ablated_neurons_across_everything = torch.zeros(model.cfg.n_heads, model.cfg.d_mlp, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
# %%
for tensor_name in tensors_to_load:
    with open(FOLDER_TO_STORE_PICKLES + f"MLPs_{tensor_name}_{safe_model_name}_L{ablate_layer}.pickle", "rb") as f:
        locals()[tensor_name] = pickle.load(f)
# %% 

# # %% 
# hist(last_layer_clean_neurons.mean((-1, -2)).cpu(), title = "Mean DE of neurons in last layer")
# top_neurons = topk_of_Nd_tensor(-1 * last_layer_clean_neurons.mean((-1, -2)), 10)

# print("Neurons with lowest DE on clean runs:")
# for neuron in top_neurons:
#    print(neuron, ":", last_layer_clean_neurons[neuron].mean((-1, -2)).item())
# %%
# neuron_self_repair = last_layer_ablated_neurons - last_layer_clean_neurons

# hist((neuron_self_repair).mean((-1, -2)).cpu(), title = "Mean Self-Repair of neurons in last layer")

# top_self_repair_neurons = topk_of_Nd_tensor((neuron_self_repair).mean((-1, -2)), 10)

# print("Neurons with top self-repair:")
# for neuron in top_self_repair_neurons:
#     print(neuron, ":", (neuron_self_repair)[neuron].mean((-1, -2)).item())


# %% The ORIGINAL GRAPH
for PERCENTILE in [0.01]: #  
    num_prompts_to_consider = int(TOTAL_PROMPTS_TO_ITERATE_THROUGH * PROMPT_LEN * PERCENTILE)
    
    # Create the plot
    fig = go.Figure()
    
    for head in range(model.cfg.n_heads):
        neuron_self_repair = last_layer_ablated_neurons_across_everything[head] - last_layer_clean_neurons_across_everything[head]
        top_self_repair = topk_of_Nd_tensor(self_repair_across_everything[head], num_prompts_to_consider)
        top_DE_in_head = topk_of_Nd_tensor(direct_effects_across_everything[head], num_prompts_to_consider)
        
        top_metric = top_DE_in_head
        # AVERAGE SELF-REPAIR

        all_cumulative_percentages = []
        

        for batch, pos in top_metric:
            # Calculate total sum for each specific (batch, pos) pair
            total_sum = neuron_self_repair[:, batch, pos].sum()
            if total_sum == 0:
                continue

            # Get the indices that would sort the array in descending order (using absolute values)
            sorted_indices = torch.argsort(neuron_self_repair[:, batch, pos].abs(), descending=True)
            #sorted_indices = torch.argsort(neuron_self_repair[:, batch, pos], descending=False)

            # Calculate the cumulative sum of the sorted tensor
            cumulative_sums = torch.cumsum(neuron_self_repair[:, batch, pos][sorted_indices], dim=0)

            # Convert to percentage of total sum
            cumulative_percentages = 100. * cumulative_sums / total_sum

            # Append the cumulative percentages to the list
            all_cumulative_percentages.append(cumulative_percentages.cpu().numpy())

        # Convert the list of numpy arrays into a 2D numpy array for easier averaging
        all_cumulative_percentages = np.array(all_cumulative_percentages)
        
        # Calculate the average cumulative percentages across all (batch, pos) pairs
        average_cumulative_percentages = np.mean(all_cumulative_percentages, axis=0)
        
        
        

        # Determine X values (number of indices)
        x_values = np.arange(1, len(neuron_self_repair[:, batch, pos]) + 1)

    
        fig.add_trace(go.Scatter(x=x_values, y=average_cumulative_percentages,
                            mode='lines', name='L' + str(ablate_layer) + 'H' + str(head), text = [str(i) for i in top_metric]))
    
    fig.update_layout(
        title=f"Average Cumulative Percentage of L{model.cfg.n_layers - 1} Self-Repair explained by Top-X Neurons on Top {PERCENTILE * 100}% Examples of L{ablate_layer}",
        xaxis_title="Top X Neurons",
        yaxis_title="Average Percentage of Total Sum",
        yaxis=dict(tickformat=".2f"),
    )

    if in_notebook_mode:
        fig.show()

    #fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"average_cumulative_self_repair_{safe_model_name}_L{ablate_layer}_{PERCENTILE}.html")


    

    # print(top_self_repair_by_last_layer)
    # print(sum([i in top_DE_in_head for i in top_self_repair]))
    # print(sum([i in top_DE_in_head for i in top_self_repair_by_last_layer]))
    # t
# %% NEWER GRAPH GRAPH

portion_of_self_repair_explained_by_layer = self_repair_from_layers_across_everything / direct_effects_across_everything
PERCENTILE = 0.01 # what percentile to filter direct effects for
num_prompts_to_consider = int(TOTAL_PROMPTS_TO_ITERATE_THROUGH * (PROMPT_LEN - 1) * PERCENTILE)
num_k_neuron = 3 # how many neurons to consider

# Create the plot
fig = go.Figure()


for head in range(model.cfg.n_heads):
    layer_importance = portion_of_self_repair_explained_by_layer[head] # dictates 'importance' of instance
    neuron_self_repair = last_layer_ablated_neurons_across_everything[head] - last_layer_clean_neurons_across_everything[head]
  
    
    indicies_to_select_in = topk_of_Nd_tensor(direct_effects_across_everything[head], num_prompts_to_consider)
    
    selected_layer_importances = []
    selected_neuron_self_repair = []
    
    
    for batch, pos in indicies_to_select_in:
        selected_layer_importances.append(layer_importance[batch, pos].item())
        selected_neuron_self_repair.append(neuron_self_repair[:, batch, pos])    
    
    # convert back to tensors
    selected_layer_importances = torch.tensor(selected_layer_importances)
    selected_neuron_self_repair = torch.stack(selected_neuron_self_repair)
    
    
    #print(selected_layer_importances.shape)
    #print(selected_neuron_self_repair.shape)
    
    # within these indicies of significant DE, order by layer_importance
    x_values = [i / 100 for i in range(101)][::-1]
    y_values = []
    for percent_SR_layer_explains in x_values:  
        indicies = torch.where((selected_layer_importances > percent_SR_layer_explains))[0]
        
        
        #print(len(indicies))
        if len(indicies) == 0:
            print("YACK")

        sub_selected_neuron_self_repair: Float[Tensor, "prompt d_mlp"] = selected_neuron_self_repair[indicies]
        
        # find top-1 neuron for each instance
        positive_self_repair_percentages = []
        
        only_positive_neurons = torch.where(sub_selected_neuron_self_repair > 0, sub_selected_neuron_self_repair, torch.tensor(0.0))
        top_k_neuron_values, _ = torch.topk(only_positive_neurons, k=num_k_neuron, dim=1)       
        sum_positive_neurons = only_positive_neurons.sum(dim=1)
        
        percent_top_k_self_repair_explained = top_k_neuron_values.sum(dim=1) / (sum_positive_neurons + 1e-10)
        average_percent_positive_self_repair_explained = percent_top_k_self_repair_explained.mean().item()
        y_values.append(average_percent_positive_self_repair_explained)
        
    fig.add_trace(go.Scatter(x=x_values, y=y_values,
                        mode='lines', name='L' + str(ablate_layer) + 'H' + str(head)))#, text = [str(i) for i in top_metric]

fig.update_layout(
    title=f"Percent top-{num_k_neuron} neuron explains of positive change in MLP Neurons across top {PERCENTILE * 100}% of L{ablate_layer} DE",
    xaxis_title="Significance of Self-Repair (fraction of DE recovered by layer)",
    yaxis_title=f"Percent of Positive Self-Repair Explained by Top-{num_k_neuron} Neuron",
    yaxis=dict(tickformat=".2f"),
)

if in_notebook_mode:
    fig.show()

    #fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"average_cumulative_self_repair_{safe_model_name}_L{ablate_layer}_{PERCENTILE}.html")


    

    # print(top_self_repair_by_last_layer)
    # print(sum([i in top_DE_in_head for i in top_self_repair]))
    # print(sum([i in top_DE_in_head for i in top_self_repair_by_last_layer]))
    # t


# %% investigate individual instances
instance = 150
head = 11
neuron_self_repair = last_layer_ablated_neurons_across_everything[head] - last_layer_clean_neurons_across_everything[head]

num_prompts_to_consider = int(TOTAL_PROMPTS_TO_ITERATE_THROUGH * PROMPT_LEN * 0.01)
top_self_repair = topk_of_Nd_tensor(self_repair_across_everything[head], num_prompts_to_consider)
top_DE_in_head = topk_of_Nd_tensor(direct_effects_across_everything[head], num_prompts_to_consider)

top_metric = top_DE_in_head

batch = top_metric[instance][0]
pos = top_metric[instance][1]



print("Total layer self-repair:",  neuron_self_repair[:, batch, pos].sum())
print("Total self-repair:", self_repair_across_everything[head, batch, pos])
print("DE:", direct_effects_across_everything[head, batch, pos])


x = last_layer_clean_neurons_across_everything[head, :, batch, pos].cpu().numpy()
y = last_layer_ablated_neurons_across_everything[head, :, batch, pos].cpu().numpy()
labels = [str(i) for i in range(model.cfg.d_mlp)]  

scatter_plot = go.Scatter(x=x, y=y, mode='markers', name='Clean vs Corrupted', text = labels)

# Add y=x line
line = go.Scatter(x=[min(x), max(x)], y=[min(x), max(x)], mode='lines', name='y=x')

# Combine data
data = [scatter_plot, line]

# Creating figure and adding data and layout
fig = go.Figure(data=data, layout=go.Layout(
    title=f"Clean vs Corrupted DE of Last Layer Neurons on B{batch}P{pos}",
    xaxis=dict(title='Clean Last Layer DE'),
    yaxis=dict(title='Corrupted Last Layer DE'),
    showlegend=True
))

if in_notebook_mode:
    fig.show()
    
# %%
# CUMULATIVE SELF-REPAIR FROM TOP-X NEURONS
total_sum = neuron_self_repair[:, batch, pos].sum()

# Get the indices that would sort the array in descending order
sorted_indices = torch.argsort(neuron_self_repair[:, batch, pos].abs(), descending=True)
# Calculate the cumulative sum of the sorted tensor
cumulative_sums = torch.cumsum(neuron_self_repair[:, batch, pos][sorted_indices], dim=0)
# Convert to percentage of total sum
cumulative_percentages = 100. * cumulative_sums / total_sum

# Determine X values (number of indices)
x_values = torch.arange(1, len(neuron_self_repair[:, batch, pos]) + 1)

# Create the plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=x_values.cpu().numpy(), y=cumulative_percentages.cpu().numpy(),
                        mode='lines', name='Cumulative Percentage'))
fig.update_layout(
    title="Cumulative Percentage of Total Self-Repair explained by Top-X Neurons",
    xaxis_title="Top X Neurons",
    yaxis_title="Percentage of Total Sum",
    yaxis=dict(tickformat=".2f"),
)

if in_notebook_mode:
    fig.show()



# %%
# # %% Generate graph
# fig = make_subplots(rows=model.cfg.n_layers, cols=1, subplot_titles=[f'Layer {l}' for l in range(model.cfg.n_layers)])

# for layer in range(model.cfg.n_layers):
#     # Add bars to the subplot for the current layer
#     x = [f'L{layer}H{h}' for h in range(model.cfg.n_heads)]
#     fig.add_trace(go.Bar(name='Heads', x=x, y=condensed_self_repair_from_heads[layer], marker_color = "red", offsetgroup=0), row=layer + 1, col=1)
#     fig.add_trace(go.Bar(name='Layers', x=x, y=condensed_self_repair_from_layers[layer], marker_color = "blue", offsetgroup=1), row=layer + 1, col=1)
#     fig.add_trace(go.Bar(name='LayerNorm', x=x, y=condensed_self_repair_from_LN[layer], marker_color = "orange", offsetgroup=2), row=layer + 1, col=1)
#     fig.add_trace(go.Bar(name='Direct Effect', x=x, y=condensed_direct_effects[layer], marker_color = "pink", offsetgroup=3), row=layer + 1, col=1)
#     fig.add_trace(go.Bar(name='Ablated Direct Effect', x=x, y=condensed_ablated_direct_effects[layer],marker_color = "purple", offsetgroup=4), row = layer + 1, col = 1)

# fig.update_layout(
#     barmode='group',
#     title=f'Self-Repair Distribution by Component | {model_name}',
#     xaxis_title='Heads',
#     yaxis_title='Average Self-Repair',
#     legend_title='Components',
#     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
#     height=300 * model.cfg.n_layers,  # Adjust the height based on the number of layers
# )
# if in_notebook_mode:
#     fig.show()
# # %%

# # save fig to html
# fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"self_repair_breakdown_{safe_model_name}.html")

# # %%

# %%
