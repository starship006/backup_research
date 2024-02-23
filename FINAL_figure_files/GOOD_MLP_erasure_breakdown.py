"""
This code breaks down self-repair in the MLP layers by neuron, and tries to understand the nature behind them.

Some interesting questions:
- Is it sparse? (i.e. are only a few neurons responsible for self-repair?)
- Is it consistent? (i.e. do the same neurons always self-repair?)
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
    ABLATE_HEAD = 8
    ablate_layer = 7 # second to last layer
    min_tokens = 100_000
    
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)  
    parser.add_argument('--head', type=int, default=9)
    parser.add_argument('--ablate_layer', type=int, default=10)
    parser.add_argument('--min_tokens', type=int, default=100_000)
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    ABLATE_HEAD = args.head
    ablate_layer = args.ablate_layer
    min_tokens = args.min_tokens
    
    


# %%
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
model.set_use_attn_result(True)
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
    
    
    
    
    return logit_diffs, direct_effects, ablated_direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN, self_repair, ablated_cache, ablated_logits
        
# %% We need to iterate through the dataset to find the 
#TOTAL_PROMPTS_TO_ITERATE_THROUGH = 3000
PROMPT_LEN = 128
PADDING = False
TOTAL_TOKENS = ((min_tokens// (PROMPT_LEN * BATCH_SIZE)) + 1) * (PROMPT_LEN * BATCH_SIZE)
dataset, num_batches = prepare_dataset(model, device, TOTAL_TOKENS, BATCH_SIZE, PROMPT_LEN, PADDING, "pile")
# %%
logit_diffs_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
direct_effects_across_everything = torch.zeros(model.cfg.n_heads,num_batches, BATCH_SIZE, PROMPT_LEN - 1)
ablated_direct_effects_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
self_repair_from_heads_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
self_repair_from_layers_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
direct_effects_from_layers_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)



self_repair_from_LN_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
self_repair_across_everything = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)


last_layer_clean_neurons_across_everything = torch.zeros(model.cfg.n_heads, model.cfg.d_mlp, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
last_layer_ablated_neurons_across_everything = torch.zeros(model.cfg.n_heads, model.cfg.d_mlp, num_batches, BATCH_SIZE, PROMPT_LEN - 1)

positive_changes_in_neuron_from_layer = torch.zeros(model.cfg.n_heads, num_batches, BATCH_SIZE, PROMPT_LEN - 1)
# %%
pbar = tqdm(total=num_batches, desc='Processing batches')
ablate_head = ABLATE_HEAD

for batch_idx, clean_tokens, corrupted_tokens in dataset:
    assert clean_tokens.shape == corrupted_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
    
    # Cache clean/corrupted model activations + direct effects
    logits, cache = model.run_with_cache(clean_tokens)
    per_head_direct_effect, all_layer_direct_effect, per_neuron_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = True)

    logit_diffs, direct_effects, ablated_direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN, self_repair, ablated_cache, ablated_logits = ablate_instance_and_get_repair((ablate_layer, ablate_head), clean_tokens, corrupted_tokens, per_head_direct_effect, all_layer_direct_effect, cache, logits)    
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect, ablated_per_neuron_direct_effect = collect_direct_effect(ablated_cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = True, cache_for_scaling = cache)
    
    # get last layer activations
    last_layer_clean_neurons: Float[Tensor, "d_mlp batch pos_minus_one"] = per_neuron_direct_effect[-1, :, :, :]
    last_layer_ablated_neurons: Float[Tensor, "d_mlp batch pos_minus_one"] = ablated_per_neuron_direct_effect[-1, :, :, :]
    
    
    # store activations
    logit_diffs_across_everything[ablate_head, batch_idx, :, :] = logit_diffs
    direct_effects_across_everything[ablate_head, batch_idx, :, :] = direct_effects
    ablated_direct_effects_across_everything[ablate_head, batch_idx, :, :] = ablated_direct_effects
    self_repair_from_heads_across_everything[ablate_head, batch_idx, :, :] = self_repair_from_heads
    self_repair_from_layers_across_everything[ablate_head, batch_idx, :, :] = self_repair_from_layers
    direct_effects_from_layers_across_everything[ablate_head, batch_idx, :, :] = all_layer_direct_effect[-1]
    
    
    self_repair_from_LN_across_everything[ablate_head, batch_idx, :, :] = self_repair_from_LN
    self_repair_across_everything[ablate_head, batch_idx, :, :] = self_repair
    
    last_layer_clean_neurons_across_everything[ablate_head, :, batch_idx, :, :] = last_layer_clean_neurons
    last_layer_ablated_neurons_across_everything[ablate_head, :, batch_idx, :, :] = last_layer_ablated_neurons

    positive_changes = (last_layer_ablated_neurons - last_layer_clean_neurons).clamp(min=0)
    positive_changes_in_neuron_from_layer[ablate_head, batch_idx, :, :] = positive_changes.sum(dim=0)
    pbar.update(1)
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
# %% Flatten the batches into one big batch

logit_diffs_across_everything = logit_diffs_across_everything.flatten(1,2)
direct_effects_across_everything = direct_effects_across_everything.flatten(1,2)
ablated_direct_effects_across_everything = ablated_direct_effects_across_everything.flatten(1,2)
self_repair_from_heads_across_everything = self_repair_from_heads_across_everything.flatten(1,2)
self_repair_from_layers_across_everything = self_repair_from_layers_across_everything.flatten(1,2)
self_repair_from_LN_across_everything = self_repair_from_LN_across_everything.flatten(1,2)
self_repair_across_everything = self_repair_across_everything.flatten(1,2)
direct_effects_from_layers_across_everything = direct_effects_from_layers_across_everything.flatten(1,2)


last_layer_clean_neurons_across_everything = last_layer_clean_neurons_across_everything.flatten(2,3)
last_layer_ablated_neurons_across_everything = last_layer_ablated_neurons_across_everything.flatten(2,3)

positive_changes_in_neuron_from_layer = positive_changes_in_neuron_from_layer.flatten(1,2)

# %%
last_layer_self_repair_across_everything = last_layer_ablated_neurons_across_everything - last_layer_clean_neurons_across_everything
# %%
sorted_neurons_across_everything = torch.sort(last_layer_self_repair_across_everything, dim = 1, descending = True)
sorted_values, neuron_indicies = sorted_neurons_across_everything
# %% Filter for top 2 % of self-repairing, and see how much the 
PERCENTILE = 0.02
num_heads, num_prompts, prompt_len_minus_one = logit_diffs_across_everything.shape
num_prompts_to_consider = int(num_prompts * prompt_len_minus_one * PERCENTILE)

positions_to_filter = []
for head in range(model.cfg.n_heads):
    direct_effect_for_head = direct_effects_across_everything[head]
    filtered_batch_pos = topk_of_Nd_tensor(direct_effect_for_head, num_prompts_to_consider)
    positions_to_filter.append(filtered_batch_pos)
    

# %% Quickly visualize ratio between positive changes and all changes
ratio = (positive_changes_in_neuron_from_layer / last_layer_self_repair_across_everything.sum(1))

# value for how much positive is outdone by other, negative forces?

filtered_ratio = [ratio[ABLATE_HEAD, batch, pos] for (batch, pos) in positions_to_filter[ABLATE_HEAD]]
print("Median is like around: ", torch.stack(filtered_ratio).nanmedian())


# %% Get ratio between top neuron, and positive change
top_x_neurons = model.cfg.d_mlp // 100
num_prompts_to_consider = int(num_prompts * prompt_len_minus_one * PERCENTILE)
print("Considering top", num_prompts_to_consider, "prompts")


to_compare_to = positive_changes_in_neuron_from_layer#last_layer_self_repair_across_everything.sum(1)

#for head in range(model.cfg.n_heads):

# get top self-repairing neurons
sums_of_top_neurons = torch.zeros(num_prompts_to_consider)
comparison = torch.zeros(num_prompts_to_consider)
for i, pos_tuple in enumerate(positions_to_filter[ABLATE_HEAD]):
    batch, pos = pos_tuple
    sums_of_top_neurons[i] = sorted_values[ABLATE_HEAD, :top_x_neurons, batch, pos].sum()
    comparison[i] = to_compare_to[ABLATE_HEAD, batch, pos]

ratio = (sums_of_top_neurons / comparison)


# Create the figure object
fig = go.Figure()

trace = go.Histogram(
    x=ratio,
    name='L' + str(ablate_layer) + 'H' + str(ABLATE_HEAD),
    opacity=0.75,
)

fig.add_trace(trace)
fig.update_xaxes(range=[0, 1])
fig.update_layout(
    title=f"Ratio of Top {top_x_neurons} Neurons to Positive Changes in L{ablate_layer}H{ABLATE_HEAD}",
)



# Display the figure
if in_notebook_mode:
    fig.show()


# %% How many times does the top X neuron explain 5% of the direct effect of the head
filtered_direct_effects = torch.stack([direct_effects_across_everything[ABLATE_HEAD, batch, pos] for (batch, pos) in positions_to_filter[ABLATE_HEAD]])
filtered_sorted_neurons = torch.stack([sorted_values[ABLATE_HEAD, :, batch, pos] for (batch, pos) in positions_to_filter[ABLATE_HEAD]])

repeated_de = einops.repeat(filtered_direct_effects, "batch -> batch neuron", neuron = model.cfg.d_mlp)


# %%
fig = go.Figure()

percentages = [0.5, 0.1, 0.05, 0.025, 0.01]
is_larger_tensors = []

for percent in percentages:
    is_larger = filtered_sorted_neurons > (repeated_de * percent)
    is_larger = (is_larger.sum(0)) / (is_larger.shape[0])
    is_larger_tensors.append(is_larger)

    fig.add_trace(go.Scatter(x=np.arange(1, model.cfg.d_mlp + 1),
                             y=is_larger,
                             mode='lines+markers',
                             name=f"{percent * 100}%",
                             ))

fig.update_layout(
    xaxis=dict(title="Top Nth Neuron"),
    yaxis=dict(title="% of instances", tickformat=".0%"),  # Specify tickformat for percentage
    height=450,
    width=600,
    margin=dict(l=50, r=50, t=50, b=50),
)
fig.update_xaxes(range=[0, 50])

if in_notebook_mode:
    fig.show()

fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"top_neuron_explains_instances_{safe_model_name}_L{ablate_layer}_H{ABLATE_HEAD}.pdf")

# %%








# %% SAVE THESE TENSORS:
tensors_to_save = {
    "is_larger_tensors": is_larger_tensors,
}

FOLDER_TO_STORE_PICKLES = "pickle_storage/mlp_sparsity/"
subfolder_path = Path(FOLDER_TO_STORE_PICKLES) / safe_model_name
subfolder_path.mkdir(parents=True, exist_ok=True)  # This creates the subfolder if it doesn't exist


# Loop through the dictionary and save each tensor
for tensor_name, tensor_data in tensors_to_save.items():
    file_path = subfolder_path / f"MLPs_{tensor_name}_{safe_model_name}_L{ablate_layer}_H{ABLATE_HEAD}.pickle"
    with open(file_path, "wb") as f:
        pickle.dump(tensor_data, f)


        
# %%




# %% Filter out neurons


N = 20  # Example value

# Initialize the result tensors with the appropriate sizes
num_heads, d_mlp, num_prompts, prompt_len = last_layer_self_repair_across_everything.shape
top_neurons_idx = torch.zeros((num_heads, num_prompts, prompt_len, N))
top_neuron_vals = torch.zeros((num_heads, num_prompts, prompt_len, N))
top_neuron_initial_vals = torch.zeros((num_heads, num_prompts, prompt_len, N))

# Iterate over each head, prompt, and position to find the top N neurons
for head in range(num_heads):
    for prompt in range(num_prompts):
        for position in range(prompt_len):
            # Select the slice for the current head, prompt, and position
            neurons = last_layer_self_repair_across_everything[head, :, prompt, position]
            
            # Use torch.topk to get the top N values and their indices
            values, indices = torch.topk(neurons, N)
            
            # Store the results in the corresponding tensors
            top_neurons_idx[head, prompt, position] = indices
            top_neuron_vals[head, prompt, position] = values
            top_neuron_initial_vals[head, prompt, position] = last_layer_clean_neurons_across_everything[head, :, prompt, position][indices]




# # %% SAVE THESE TENSORS:
# tensors_to_save = {
#     "logit_diffs_across_everything": logit_diffs_across_everything,
#     "direct_effects_across_everything": direct_effects_across_everything,
#     "ablated_direct_effects_across_everything": ablated_direct_effects_across_everything,
#     "self_repair_from_heads_across_everything": self_repair_from_heads_across_everything,
#     "self_repair_from_layers_across_everything": self_repair_from_layers_across_everything,
#     "self_repair_from_LN_across_everything": self_repair_from_LN_across_everything,
#     "self_repair_across_everything": self_repair_across_everything,
#     #"last_layer_clean_neurons_across_everything": last_layer_clean_neurons_across_everything,
#     #"last_layer_ablated_neurons_across_everything": last_layer_ablated_neurons_across_everything,
#     "top_neurons_idx": top_neurons_idx,
#     "top_neuron_vals": top_neuron_vals,
#     "top_neuron_initial_vals": top_neuron_initial_vals,
#     "positive_changes_in_neuron_from_layer": positive_changes_in_neuron_from_layer,
#     "direct_effects_from_layers_across_everything": direct_effects_from_layers_across_everything,
# }

# FOLDER_TO_STORE_PICKLES = "pickle_storage/mlp_sparsity/"
# subfolder_path = Path(FOLDER_TO_STORE_PICKLES) / safe_model_name
# subfolder_path.mkdir(parents=True, exist_ok=True)  # This creates the subfolder if it doesn't exist


# # Loop through the dictionary and save each tensor
# for tensor_name, tensor_data in tensors_to_save.items():
#     file_path = subfolder_path / f"MLPs_{tensor_name}_{safe_model_name}_L{ablate_layer}.pickle"
#     with open(file_path, "wb") as f:
#         pickle.dump(tensor_data, f)

# %%








# # %% First: MLP explains significant portion of self-repair

# ADD_BOUNDS = True # if we should bound the values because noise
# for PERCENTILE in [0.1]: # 0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
#     num_prompts_to_consider = int(BATCH_SIZE * num_batches * PROMPT_LEN * PERCENTILE)
#     print("Considering top", num_prompts_to_consider, "prompts")
#     # Create the plot
#     fig = go.Figure()
    
#     for head in range(model.cfg.n_heads):
#         neuron_self_repair = last_layer_ablated_neurons_across_everything[head] - last_layer_clean_neurons_across_everything[head]
#         total_self_repair = self_repair_across_everything[head]
        
#         top_DE_in_head = topk_of_Nd_tensor(direct_effects_across_everything[head], num_prompts_to_consider)
#         filtered_batch_pos = top_DE_in_head
        
#         # AVERAGE SELF-REPAIR
#         all_cumulative_percentages = []
#         for batch, pos in filtered_batch_pos:
#             summed_neuron_self_repair_instance = max(neuron_self_repair[:, batch, pos].sum(), 0) # if negative, make zero
#             self_repair_instance = total_self_repair[batch, pos]
#             if self_repair_instance <= 0:
#                 continue
            
            
#             percentage_from_neurons = (100. * summed_neuron_self_repair_instance / self_repair_instance).item()
            
#             if ADD_BOUNDS:
#                 percentage_from_neurons = min(percentage_from_neurons, 100)
#                 percentage_from_neurons = max(percentage_from_neurons, 0)
            
#             all_cumulative_percentages.append(percentage_from_neurons)
            
#         all_cumulative_percentages = np.array(all_cumulative_percentages)
#         average_cumulative_percentages = np.mean(all_cumulative_percentages, axis=0)
        
#         print(f"Head {head} | len {len(all_cumulative_percentages)} : " + str(average_cumulative_percentages))
#         px.histogram(all_cumulative_percentages, title = f"Head {head} | len {len(all_cumulative_percentages)}").show()
        
        
        
#         # # Determine X values (number of indices)
#         # x_values = np.arange(1, len(neuron_self_repair[:, batch, pos]) + 1)

    
#         # fig.add_trace(go.Scatter(x=x_values, y=average_cumulative_percentages,
#         #                     mode='lines', name='L' + str(ablate_layer) + 'H' + str(head), text = [str(i) for i in filtered_batch_pos]))
    
#     # fig.update_layout(
#     #     title=f"Average Cumulative Percentage of L{model.cfg.n_layers - 1} Self-Repair explained by Top-X Neurons on Top {PERCENTILE * 100}% Examples of L{ablate_layer}",
#     #     xaxis_title="Top X Neurons",
#     #     yaxis_title="Average Percentage of Total Sum",
#     #     yaxis=dict(tickformat=".2f"),
#     # )

#     # if in_notebook_mode:
#     #     fig.show()

#     #fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"average_cumulative_self_repair_{safe_model_name}_L{ablate_layer}_{PERCENTILE}.html")






 # %% A few neurons explain all the ABSOLUTE CHANGE IN NEURONS
for PERCENTILE in [0.02]: # 0.001, 0.005, 0.01, 0.02, 0.05, 0.1,
    num_prompts_to_consider = int(BATCH_SIZE * num_batches * PROMPT_LEN * PERCENTILE)
    
    # Create the plot
    fig = go.Figure()
    
    neuron_self_repair = last_layer_self_repair_across_everything[ablate_head]
    
    #top_self_repair = topk_of_Nd_tensor(self_repair_across_everything[head], num_prompts_to_consider)
    #top_self_repair_by_last_layer = topk_of_Nd_tensor(self_repair_from_layers_across_everything[head], num_prompts_to_consider)
    top_DE_in_head = topk_of_Nd_tensor(direct_effects_across_everything[ablate_head], num_prompts_to_consider)
    
    filtered_batch_pos = top_DE_in_head
    
    # AVERAGE SELF-REPAIR
    all_cumulative_percentages = []
    all_des_of_neurons = []
    how_many_huh = 0
    for batch, pos in filtered_batch_pos:
        # Calculate total sum for each specific (batch, pos) pair
        
        #total_neuron_self_repair = neuron_self_repair[:, batch, pos].sum()
        absolute_changes = neuron_self_repair[:, batch, pos].abs()
        summed_absolute_change = absolute_changes.sum()
        
        if neuron_self_repair[:, batch, pos].sum() < 0.1:
            print("Huh", neuron_self_repair[:, batch, pos].sum())
            how_many_huh += 1
            
             

        # Get the indices that would sort the array in descending order (using absolute values)
        sorted_indices = torch.argsort(neuron_self_repair[:, batch, pos], descending=True)

        # Calculate the cumulative sum of absolute changes of the sorted tensor
        cumulative_sums = torch.cumsum(absolute_changes[sorted_indices], dim=0)

        # Convert to percentage of total sum
        cumulative_percentages = 100. * cumulative_sums / summed_absolute_change

        # Append the cumulative percentages to the list
        all_cumulative_percentages.append(cumulative_percentages.cpu().numpy())
        
        # Get the initial direct effect of the neurons
        all_des_of_neurons.append(last_layer_clean_neurons_across_everything[ablate_head, :, batch, pos][sorted_indices])
        
    print(how_many_huh)
    # Convert the list of numpy arrays into a 2D numpy array for easier averaging
    
    all_cumulative_percentages = np.array(all_cumulative_percentages)
    all_des_of_neurons = np.array(all_des_of_neurons)
    
    
    # Calculate the average cumulative percentages across all (batch, pos) pairs
    average_cumulative_percentages = np.mean(all_cumulative_percentages, axis=0)
    avg_all_des_of_neurons = np.median(all_des_of_neurons, axis=0)
    

    # Determine X values (number of indices)
    x_values = np.arange(1, len(neuron_self_repair[:, batch, pos]) + 1)

    # Add trace, with the color of each dot being the de of the neuron
    fig.add_trace(go.Scatter(x=x_values, y=average_cumulative_percentages,
                        mode='markers', 
                        marker=dict(
                            color=avg_all_des_of_neurons,
                            colorscale=['red', 'white', 'blue'],
                            colorbar=dict(
                                title='Median Clean DE',
                                #tickvals=[0, num_layers - 1],
                                #ticktext=[0, 1,2,1,1,1,1,1,1,1,1,1,1,11,1,1,3,4,5,5,num_layers - 1],
                                #orientation="h"
                            ),
                            cmid = 0
                            ),
                        name='L' + str(ablate_layer) + 'H' + str(head), text = [str(i) for i in filtered_batch_pos],))
                  
    fig.update_layout(
        #title=f"Average Cumulative Percentage of L{model.cfg.n_layers - 1} Self-Repair explained by Top-X Neurons on Top {PERCENTILE * 100}% Examples of L{ablate_layer}",
        xaxis_title="Top Xth Self-Repairing Neuron",
        yaxis_title="Cumulative Percentage of Absolute Change",
        yaxis=dict(tickformat=".2f"),
    )

    if in_notebook_mode:
        fig.show()

    fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"average_cumulative_self_repair_{safe_model_name}_L{ablate_layer}_H{ablate_head}_{PERCENTILE}.html")
    fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"average_cumulative_self_repair_{safe_model_name}_L{ablate_layer}_H{ablate_head}_{PERCENTILE}.pdf")


# %%
# %% Save the data to pickle
# tensors_to_save = {
#     "last_layer_clean_neurons_across_everything": last_layer_clean_neurons_across_everything,
#     "last_layer_ablated_neurons_across_everything": last_layer_ablated_neurons_across_everything,
#     "self_repair_across_everything": self_repair_across_everything,
#     "direct_effects_across_everything": direct_effects_across_everything,
#     "self_repair_from_layers_across_everything": self_repair_from_layers_across_everything,
# }

# # Loop through the dictionary and save each tensor
# for tensor_name, tensor_data in tensors_to_save.items():
#     with open(FOLDER_TO_STORE_PICKLES + f"MLPs_{tensor_name}_{safe_model_name}_L{ablate_layer}.pickle", "wb") as f:
#         pickle.dump(tensor_data, f)

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
