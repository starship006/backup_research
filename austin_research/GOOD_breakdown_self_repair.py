"""
This code breaks down self-repair into three components:
  - LN
  - attn heads
  - mlp layers
  
And determines how much each component contributes to self-repair
"""
# %%
from imports import *
# %%

from path_patching import act_patch
from GOOD_helpers import *
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "pythia-410m"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name

BATCH_SIZE = 60
PROMPT_LEN = 40

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
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_owt_tokens)
# %%
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=owt_tokens, model = model, display = in_notebook_mode, collect_individual_neurons = False)
if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")
# %%
def ablate_top_instances(head: tuple, top_instances_count = 40):    
    top_indicies = topk_of_Nd_tensor(per_head_direct_effect[head[0],head[1]], top_instances_count + 1)

    ablated_cache = act_patch(model, owt_tokens, [Node("z", head[0], head[1])], return_item, corrupted_owt_tokens, apply_metric_to_cache = True)
    ablated_logits = act_patch(model, owt_tokens, [Node("z", head[0], head[1])], return_item, corrupted_owt_tokens, apply_metric_to_cache = False)
        
    FREEZE_FINAL_LN = True
    if FREEZE_FINAL_LN:
        ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(ablated_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
    else:
        ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(ablated_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False)


    logit_diffs = torch.zeros(top_instances_count)
    direct_effects = torch.zeros(top_instances_count)
    ablated_direct_effects = torch.zeros(top_instances_count)
    self_repair_from_heads = torch.zeros(top_instances_count)
    self_repair_from_layers = torch.zeros(top_instances_count)
    self_repair_from_LN = torch.zeros(top_instances_count)
    
    
    for instance in range(top_instances_count):
        top_index = top_indicies[instance]
        batch = top_index[0]
        pos = top_index[1]
        
        clean_logit = get_single_correct_logit(logits, batch, pos, owt_tokens)
        ablated_logit = get_single_correct_logit(ablated_logits, batch, pos, owt_tokens)
        
        logit_diff = ablated_logit - clean_logit
        orig_direct_effect = per_head_direct_effect[head[0],head[1],batch,pos]
        new_direct_effect = ablated_per_head_direct_effect[head[0],head[1],batch,pos]
        change_direct_effect = new_direct_effect - orig_direct_effect
        
        
        self_repair = logit_diff - change_direct_effect
        change_in_all_heads = (ablated_per_head_direct_effect - per_head_direct_effect)[:, :, batch, pos].sum()
        self_repair_from_heads_on_instance = change_in_all_heads - (ablated_per_head_direct_effect - per_head_direct_effect)[head[0],head[1],batch,pos]        
        self_repair_from_layers_on_instance = (ablated_all_layer_direct_effect - all_layer_direct_effect)[:, batch, pos].sum()
        
        
        self_repair_from_LN_on_instance = self_repair - self_repair_from_heads_on_instance - self_repair_from_layers_on_instance
        
        logit_diffs[instance] = logit_diff
        direct_effects[instance] = orig_direct_effect
        ablated_direct_effects[instance] = new_direct_effect
        self_repair_from_heads[instance] = self_repair_from_heads_on_instance
        self_repair_from_layers[instance] = self_repair_from_layers_on_instance
        self_repair_from_LN[instance] = self_repair_from_LN_on_instance
        
        
    return logit_diffs, direct_effects, ablated_direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN
        
        
        
#logit_diffs, direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN = ablate_top_instances((22,11))
        
# %%

n_layers = model.cfg.n_layers
fig = make_subplots(rows=n_layers, cols=1, subplot_titles=[f'Layer {l}' for l in range(n_layers)])

for layer in tqdm(range(n_layers)):
    avg_heads = []
    avg_layers = []
    avg_LN = []
    avg_direct_effects = []
    avg_ablated_direct_effects = []
    x = []
    

    for head in range(model.cfg.n_heads):
        logit_diffs, direct_effects, ablated_direct_effects, self_repair_from_heads, self_repair_from_layers, self_repair_from_LN = ablate_top_instances((layer, head))
        
        avg_heads.append(self_repair_from_heads.mean())
        avg_layers.append(self_repair_from_layers.mean())
        avg_LN.append(self_repair_from_LN.mean())
        avg_direct_effects.append(direct_effects.mean())
        avg_ablated_direct_effects.append(ablated_direct_effects.mean())
        x.append(f"H{head}")

    # Add bars to the subplot for the current layer
    fig.add_trace(go.Bar(name='Heads', x=x, y=avg_heads, marker_color = "red", offsetgroup=0), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='Layers', x=x, y=avg_layers, marker_color = "blue", offsetgroup=1), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='LayerNorm', x=x, y=avg_LN, marker_color = "orange", offsetgroup=2), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='Direct Effect', x=x, y=avg_direct_effects, marker_color = "pink", offsetgroup=3), row=layer + 1, col=1)
    fig.add_trace(go.Bar(name='Ablated Direct Effect', x=x, y=avg_ablated_direct_effects,marker_color = "purple", offsetgroup=4), row = layer + 1, col = 1)

# %%

fig.update_layout(
    barmode='group',
    title=f'Self-Repair Distribution by Component | {model_name}',
    xaxis_title='Heads',
    yaxis_title='Average Self-Repair',
    legend_title='Components',
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=300 * n_layers,  # Adjust the height based on the number of layers
)
if in_notebook_mode:
    fig.show()
# %%

# save fig to html
fig.write_html(f"self_repair_breakdown_{safe_model_name}.html")

# %%
