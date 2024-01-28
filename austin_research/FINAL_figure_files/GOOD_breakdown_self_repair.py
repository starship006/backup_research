"""
This code breaks down self-repair into three components:
  - LN
  - attn heads
  - mlp layers
  
And determines how much each component contributes to self-repair
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, show_input, collect_direct_effect, get_single_correct_logit, topk_of_Nd_tensor, return_item, get_correct_logit_score, prepare_dataset
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/breakdown_self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/breakdown_self_repair/"


if in_notebook_mode:
    model_name = "pythia-160m"
    BATCH_SIZE = 2
    PERCENTILE = 0.02
    MIN_TOKENS = 1_000
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)  
    parser.add_argument('--percentile', type=float, default=0.02)
    parser.add_argument('--min_tokens', type=int, default=100_000)
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    PERCENTILE = args.percentile
    MIN_TOKENS = args.min_tokens
    
    


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
dataset = utils.get_dataset("pile")
dataset_name = "The Pile"
all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
# %% 

def ablate_top_instances_and_get_breakdown(head: tuple, clean_tokens: Tensor, corrupted_tokens: Tensor,
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
        per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
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
PROMPT_LEN = 128
TOTAL_TOKENS = ((MIN_TOKENS // (PROMPT_LEN * BATCH_SIZE)) + 1) * (PROMPT_LEN * BATCH_SIZE)


dataset, num_batches = prepare_dataset(model, device, TOTAL_TOKENS, BATCH_SIZE, PROMPT_LEN, False, "pile")
TOTAL_PROMPTS_TO_ITERATE_THROUGH = num_batches * BATCH_SIZE
# %%
logit_diffs_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
direct_effects_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
ablated_direct_effects_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
self_repair_from_heads_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
self_repair_from_layers_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
self_repair_from_LN_across_everything = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)


percent_LN_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
percent_heads_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
percent_layers_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
percent_self_repair_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)

unclipped_percent_LN_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
unclipped_percent_heads_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
unclipped_percent_layers_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)
unclipped_percent_self_repair_of_DE = torch.zeros(TOTAL_PROMPTS_TO_ITERATE_THROUGH, PROMPT_LEN - 1, model.cfg.n_layers, model.cfg.n_heads)

# %%
pbar = tqdm(total=num_batches, desc='Processing batches')

for batch_idx, clean_tokens, corrupted_tokens in dataset:
    assert clean_tokens.shape == corrupted_tokens.shape == (BATCH_SIZE, PROMPT_LEN)
    start_clean_prompt = batch_idx * BATCH_SIZE
    end_clean_prompt = start_clean_prompt + BATCH_SIZE
    
    # Cache clean/corrupted model activations + direct effects
    logits, cache = model.run_with_cache(clean_tokens)
    per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False)

    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            logit_diffs, direct_effects, ablated_direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN, self_repair = ablate_top_instances_and_get_breakdown((layer, head), clean_tokens, corrupted_tokens, per_head_direct_effect, all_layer_direct_effect, cache, logits)
            
            logit_diffs_across_everything[start_clean_prompt:end_clean_prompt, :, layer, head] = logit_diffs
            direct_effects_across_everything[start_clean_prompt:end_clean_prompt, :, layer, head] = direct_effects
            ablated_direct_effects_across_everything[start_clean_prompt:end_clean_prompt, :, layer, head] = ablated_direct_effects
            self_repair_from_heads_across_everything[start_clean_prompt:end_clean_prompt, :, layer, head] = self_repair_from_heads
            self_repair_from_layers_across_everything[start_clean_prompt:end_clean_prompt, :, layer, head] = self_repair_from_layers
            self_repair_from_LN_across_everything[start_clean_prompt:end_clean_prompt, :, layer, head] = self_repair_from_LN
            
            percent_LN_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = np.clip((self_repair_from_LN / direct_effects).cpu(), 0, 1)
            percent_heads_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = np.clip((self_repair_from_heads / direct_effects).cpu(), 0, 1)
            percent_layers_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = np.clip((self_repair_from_layers / direct_effects).cpu(), 0, 1)
            percent_self_repair_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = np.clip((self_repair / direct_effects).cpu(), 0, 1)
            
            unclipped_percent_LN_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = (self_repair_from_LN / direct_effects).cpu()
            unclipped_percent_heads_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = (self_repair_from_heads / direct_effects).cpu()
            unclipped_percent_layers_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = (self_repair_from_layers / direct_effects).cpu()
            unclipped_percent_self_repair_of_DE[start_clean_prompt:end_clean_prompt, :, layer, head] = (self_repair / direct_effects).cpu()
            
    pbar.update(1)
        
# %%
num_top_instances = int(PERCENTILE * TOTAL_PROMPTS_TO_ITERATE_THROUGH * (PROMPT_LEN - 1))

condensed_logit_diff = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_direct_effects = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_ablated_direct_effects = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_self_repair_from_heads = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_self_repair_from_layers = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_self_repair_from_LN = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

condensed_percent_LN_of_DE = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_percent_heads_of_DE = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_percent_layers_of_DE = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
condensed_percent_self_repair_of_DE = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))


for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        # get top indicies
        top_indices = topk_of_Nd_tensor(direct_effects_across_everything[..., layer, head], num_top_instances)
        
        # find average of values over these indicies  
        condensed_logit_diff[layer, head] = torch.stack([logit_diffs_across_everything[batch, pos, layer, head] for batch, pos in top_indices]).mean()
        condensed_direct_effects[layer, head] = torch.stack([direct_effects_across_everything[batch, pos, layer, head] for batch, pos in top_indices]).mean()
        condensed_ablated_direct_effects[layer, head] = torch.stack([ablated_direct_effects_across_everything[batch, pos, layer, head] for batch, pos in top_indices]).mean()
        condensed_self_repair_from_heads[layer, head] = torch.stack([self_repair_from_heads_across_everything[batch, pos, layer, head] for batch, pos in top_indices]).mean()
        condensed_self_repair_from_layers[layer, head] = torch.stack([self_repair_from_layers_across_everything[batch, pos, layer, head] for batch, pos in top_indices]).mean()
        condensed_self_repair_from_LN[layer, head] = torch.stack([self_repair_from_LN_across_everything[batch, pos, layer, head] for batch, pos in top_indices]).mean()
        
        condensed_percent_LN_of_DE[layer, head] = torch.stack([percent_LN_of_DE[batch, pos, layer, head] for batch, pos in top_indices]).nanmean()
        condensed_percent_heads_of_DE[layer, head] = torch.stack([percent_heads_of_DE[batch, pos, layer, head] for batch, pos in top_indices]).nanmean()
        condensed_percent_layers_of_DE[layer, head] = torch.stack([percent_layers_of_DE[batch, pos, layer, head] for batch, pos in top_indices]).nanmean()
        condensed_percent_self_repair_of_DE[layer, head] = torch.stack([percent_self_repair_of_DE[batch, pos, layer, head] for batch, pos in top_indices]).nanmean()
        

# do unfiltered, too        
full_logit_diff = logit_diffs_across_everything.mean((0, 1))
full_direct_effects = direct_effects_across_everything.mean((0, 1))
full_ablated_direct_effects = ablated_direct_effects_across_everything.mean((0, 1))
full_self_repair_from_heads = self_repair_from_heads_across_everything.mean((0, 1))
full_self_repair_from_layers = self_repair_from_layers_across_everything.mean((0, 1))
full_self_repair_from_LN = self_repair_from_LN_across_everything.mean((0, 1))

full_percent_LN_of_DE = percent_LN_of_DE.nanmean((0, 1))
full_percent_heads_of_DE = percent_heads_of_DE.nanmean((0, 1))
full_percent_layers_of_DE = percent_layers_of_DE.nanmean((0, 1))
full_percent_self_repair_of_DE = percent_self_repair_of_DE.nanmean((0, 1))

full_unclipped_percent_LN_of_DE = unclipped_percent_LN_of_DE.nanmean((0, 1))
full_unclipped_percent_heads_of_DE = unclipped_percent_heads_of_DE.nanmean((0, 1))
full_unclipped_percent_layers_of_DE = unclipped_percent_layers_of_DE.nanmean((0, 1))
full_unclipped_percent_self_repair_of_DE = unclipped_percent_self_repair_of_DE.nanmean((0, 1))


# %% Save the data to pickle
tensors_to_save = {
    "condensed_logit_diff": condensed_logit_diff,
    "condensed_direct_effects": condensed_direct_effects,
    "condensed_ablated_direct_effects": condensed_ablated_direct_effects,
    "condensed_self_repair_from_heads": condensed_self_repair_from_heads,
    "condensed_self_repair_from_layers": condensed_self_repair_from_layers,
    "condensed_self_repair_from_LN": condensed_self_repair_from_LN,
    
    "condensed_percent_LN_of_DE": condensed_percent_LN_of_DE,
    "condensed_percent_heads_of_DE": condensed_percent_heads_of_DE,
    "condensed_percent_layers_of_DE": condensed_percent_layers_of_DE,
    "condensed_percent_self_repair_of_DE": condensed_percent_self_repair_of_DE,
    
    "full_logit_diff": full_logit_diff,
    "full_direct_effects": full_direct_effects,
    "full_ablated_direct_effects": full_ablated_direct_effects,
    "full_self_repair_from_heads": full_self_repair_from_heads,
    "full_self_repair_from_layers": full_self_repair_from_layers,
    "full_self_repair_from_LN": full_self_repair_from_LN,
    
    "full_percent_LN_of_DE": full_percent_LN_of_DE,
    "full_percent_heads_of_DE": full_percent_heads_of_DE,
    "full_percent_layers_of_DE": full_percent_layers_of_DE,
    "full_percent_self_repair_of_DE": full_percent_self_repair_of_DE,
    
    "full_unclipped_percent_LN_of_DE": full_unclipped_percent_LN_of_DE,
    "full_unclipped_percent_heads_of_DE": full_unclipped_percent_heads_of_DE,
    "full_unclipped_percent_layers_of_DE": full_unclipped_percent_layers_of_DE,
    "full_unclipped_percent_self_repair_of_DE": full_unclipped_percent_self_repair_of_DE,
}

percentile_str = "" if PERCENTILE == 0.02 else f"{PERCENTILE}_" # 0.02 is the default

# Loop through the dictionary and save each tensor
for tensor_name, tensor_data in tensors_to_save.items():
    with open(FOLDER_TO_STORE_PICKLES + f"{percentile_str}{safe_model_name}_{tensor_name}", "wb") as f:
        pickle.dump(tensor_data, f)
# %% Generate graph
fig = make_subplots(rows=model.cfg.n_layers, cols=1, subplot_titles=[f'Layer {l}' for l in range(model.cfg.n_layers)])

for layer in range(model.cfg.n_layers):
    # Add bars to the subplot for the current layer
    x = [f'L{layer}H{h}' for h in range(model.cfg.n_heads)]
    fig.add_trace(go.Bar(name='Heads', x=x, y=condensed_self_repair_from_heads[layer], marker_color = "red", offsetgroup=0), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='Layers', x=x, y=condensed_self_repair_from_layers[layer], marker_color = "blue", offsetgroup=1), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='LayerNorm', x=x, y=condensed_self_repair_from_LN[layer], marker_color = "orange", offsetgroup=2), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='Direct Effect', x=x, y=condensed_direct_effects[layer], marker_color = "pink", offsetgroup=3), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='Ablated Direct Effect', x=x, y=condensed_ablated_direct_effects[layer],marker_color = "purple", offsetgroup=4), row = layer + 1, col = 1)

fig.update_layout(
    barmode='group',
    title=f'Self-Repair Distribution by Component | {model_name} | {PERCENTILE}th Percentile',
    xaxis_title='Heads',
    yaxis_title='Average Self-Repair',
    legend_title='Components',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=300 * model.cfg.n_layers,  # Adjust the height based on the number of layers
)
if in_notebook_mode:
    fig.show()
# %%

# save fig to html
fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"{percentile_str}self_repair_breakdown_{safe_model_name}.html")

# %%
