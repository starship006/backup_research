"""
This code breaks down self-repair in the MLP layers by neuron, and tries to understand the nature behind them.

Some interesting questions:
- Is it sparse? (i.e. are only a few neurons responsible for self-repair?)
- Is it consistent? (i.e. do the same neurons always self-repair?)
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
TOTAL_PROMPTS_TO_ITERATE_THROUGH = 3000
PROMPT_LEN = 400

num_batches = TOTAL_PROMPTS_TO_ITERATE_THROUGH // BATCH_SIZE
# %%
logit_diffs_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
direct_effects_across_everything = torch.zeros(model.cfg.n_heads,TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
ablated_direct_effects_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
self_repair_from_heads_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
self_repair_from_layers_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
self_repair_from_LN_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
self_repair_across_everything = torch.zeros(model.cfg.n_heads, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)



last_layer_clean_neurons_across_everything = torch.zeros(model.cfg.n_heads, model.cfg.d_mlp, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
last_layer_ablated_neurons_across_everything = torch.zeros(model.cfg.n_heads, model.cfg.d_mlp, TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1)
# %%
ablate_layer = model.cfg.n_layers - 2 # second to last layer

for batch in tqdm(range(num_batches)):
    for ablate_head in range(model.cfg.n_heads):
        # Get a batch of clean and corrupted tokens
        clean_batch_offset = batch * BATCH_SIZE
        start_clean_prompt = clean_batch_offset
        end_clean_prompt = clean_batch_offset + BATCH_SIZE
        
        corrupted_batch_offset = (batch + 1) * BATCH_SIZE
        start_corrupted_prompt = corrupted_batch_offset
        end_corrupted_prompt = corrupted_batch_offset + BATCH_SIZE

        clean_tokens = all_dataset_tokens[start_clean_prompt:end_clean_prompt, :PROMPT_LEN]
        corrupted_tokens = all_dataset_tokens[start_corrupted_prompt:end_corrupted_prompt, :PROMPT_LEN]
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
        logit_diffs_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = logit_diffs
        direct_effects_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = direct_effects
        ablated_direct_effects_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = ablated_direct_effects
        self_repair_from_heads_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = self_repair_from_heads
        self_repair_from_layers_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = self_repair_from_layers
        self_repair_from_LN_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = self_repair_from_LN
        self_repair_across_everything[ablate_head, clean_batch_offset:corrupted_batch_offset, :] = self_repair
        
        last_layer_clean_neurons_across_everything[ablate_head, :, clean_batch_offset:corrupted_batch_offset, :] = last_layer_clean_neurons
        last_layer_ablated_neurons_across_everything[ablate_head, :, clean_batch_offset:corrupted_batch_offset, :] = last_layer_ablated_neurons
        
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


# %%


for PERCENTILE in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]: # 
    num_prompts_to_consider = int(TOTAL_PROMPTS_TO_ITERATE_THROUGH * PROMPT_LEN * PERCENTILE)
    
    # Create the plot
    fig = go.Figure()
    
    for head in range(model.cfg.n_heads):
        neuron_self_repair = last_layer_ablated_neurons_across_everything[head] - last_layer_clean_neurons_across_everything[head]
        top_self_repair = topk_of_Nd_tensor(self_repair_across_everything[head], num_prompts_to_consider)
        top_self_repair_by_last_layer = topk_of_Nd_tensor(self_repair_from_layers_across_everything[head], num_prompts_to_consider)
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

    fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"average_cumulative_self_repair_{safe_model_name}_L{ablate_layer}_{PERCENTILE}.html")


# %%
# %% Save the data to pickle
tensors_to_save = {
    "last_layer_clean_neurons_across_everything": last_layer_clean_neurons_across_everything,
    "last_layer_ablated_neurons_across_everything": last_layer_ablated_neurons_across_everything,
    "self_repair_across_everything": self_repair_across_everything,
    "direct_effects_across_everything": direct_effects_across_everything,
    "self_repair_from_layers_across_everything": self_repair_from_layers_across_everything,
}

# Loop through the dictionary and save each tensor
for tensor_name, tensor_data in tensors_to_save.items():
    with open(FOLDER_TO_STORE_PICKLES + f"MLPs_{tensor_name}_{safe_model_name}_L{ablate_layer}.pickle", "wb") as f:
        pickle.dump(tensor_data, f)

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
