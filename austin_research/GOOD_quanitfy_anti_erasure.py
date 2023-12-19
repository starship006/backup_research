"""
Goal: put a number behind the amount of self-repair thats explained purely by anti-erasure

This is going to pivot to just looking at self-repair due to MLP layers, who appear to have a sort of erasing behavior
"""
# %%
from imports import *
# %%
%load_ext autoreload
%autoreload 2
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
def ablate_head_freeze_erasure(head: tuple, batch = 0, pos = 0, freeze_erasure = True, declare_mlp_as_erasure: Union[list, None] = None):
    """
    for a specific batch and position, ablates a head and potentially freeze the erasure (in MLP or heads) too
    
    declare_last_mlp_as_erasure: if nonempty list, then the last MLP layer will be considered as 'erasure' and nothing else
    """
    
    model.reset_hooks()
    freeze_count = 0
    
    # add freeze erasure hooks
    if freeze_erasure:
        if declare_mlp_as_erasure is not None:
            for freeze_mlp in declare_mlp_as_erasure:
                freeze_count += 1
                frozen_output: Float[Tensor, "d_head"] = corrupted_cache[utils.get_act_name("post",  freeze_mlp)][batch, pos, :]
                freeze_hook = partial(replace_output_of_specific_MLP_batch_pos_hook, new_output = frozen_output, batch = batch, pos = pos)
                model.add_hook(utils.get_act_name("post",  freeze_mlp), freeze_hook)
        else:
            # get all the model components which contributed negatively on forward pass
            head_contribution: Float[Tensor, "layer head"] = per_head_direct_effect[..., batch, pos]
            mlp_contribution: Float[Tensor, "layer"] = all_layer_direct_effect[..., batch, pos]
            
            negative_theshold = -0.05
            negative_h_layer_index, negative_h_head_index = torch.where(head_contribution < negative_theshold)
            head_to_freeze = [(int(h_layer.item()), int(h_head.item())) for h_layer, h_head in zip(negative_h_layer_index, negative_h_head_index)]
            mlp_to_freeze = torch.where(mlp_contribution < negative_theshold)[0].tolist()
            
            
            for freeze_head in head_to_freeze:
                if freeze_head[0] > head[0]:
                    freeze_count += 1
                    frozen_output: Float[Tensor, "d_head"] = corrupted_cache[utils.get_act_name("z",  freeze_head[0])][batch, pos, freeze_head[1], :]
                    freeze_hook = partial(replace_output_of_specific_batch_pos_hook, new_output = frozen_output, head = freeze_head[1], batch = batch, pos = pos)
                    model.add_hook(utils.get_act_name("z", freeze_head[0]), freeze_hook)
                    #print(f"added hook for {freeze_head}, which has a direct effect of {head_contribution[freeze_head[0], freeze_head[1]]}")
                    
            for freeze_mlp in mlp_to_freeze:
                if freeze_mlp > head[0]:
                    freeze_count += 1
                    frozen_output: Float[Tensor, "d_head"] = corrupted_cache[utils.get_act_name("post",  freeze_mlp)][batch, pos, :]
                    freeze_hook = partial(replace_output_of_specific_MLP_batch_pos_hook, new_output = frozen_output, batch = batch, pos = pos)
                    model.add_hook(utils.get_act_name("post",  freeze_mlp), freeze_hook)
                #print("added hook")
                
                
        # print("Froze a total of ", freeze_count, "heads and MLP layers")

    # corrupt head output
    corrupted_output: Float[Tensor, "batch pos d_head"] = corrupted_cache[utils.get_act_name("z",  head[0])][...,  head[1], :]
    model.add_hook(utils.get_act_name("z", head[0]), partial(replace_output_hook, new_output = corrupted_output, head = head[1]))
    
    # run the model
    ablated_logits = model(owt_tokens)
    model.reset_hooks()
    # get new logit in direct effect compared to original
    
    correct_logit = get_single_correct_logit(ablated_logits, batch, pos, owt_tokens)
    return correct_logit, freeze_count
    
    
# %%


head = (22, 11)
TOP_INSTANCES_COUNT = 30
top_indicies = topk_of_Nd_tensor(per_head_direct_effect[head[0],head[1]], TOP_INSTANCES_COUNT + 1)

normal_diffs = []
frozen_diffs = []
for instance in range(TOP_INSTANCES_COUNT):
    top_index = top_indicies[instance]
    batch = top_index[0]
    pos = top_index[1]
    print("--------- Instance", instance, ": ", batch, pos, "---------")


    frozen_ablated_logit, freeze_count = ablate_head_freeze_erasure(head = head, batch = batch, pos = pos) 
    normal_ablated_logit, _ = ablate_head_freeze_erasure(head = head, batch = batch, pos = pos, freeze_erasure = False)

    clean_logit = get_single_correct_logit(logits, batch, pos, owt_tokens)
    print("Clean logit of token", clean_logit)
    print("Frozen logit", frozen_ablated_logit)
    print("Normal ablation logit", normal_ablated_logit)
    
    
    
    
    frozen_diff = (frozen_ablated_logit - clean_logit).item()
    normal_diff = (normal_ablated_logit - clean_logit).item()
    
    if freeze_count > 0:
        print("Anti-erasure explains", (frozen_diff - normal_diff) / (normal_diff), "of the self-repair")


    frozen_diffs.append(frozen_diff)
    normal_diffs.append(normal_diff)
# %%
# Create the scatter plot
fig = go.Figure(data=go.Scatter(x=normal_diffs, y=frozen_diffs, mode='markers'))

# Determine the range for the y=x line
min_val = min(normal_diffs)
max_val = max(normal_diffs)

# Create a line from (min_val, min_val) to (max_val, max_val)
line_x = [min_val, max_val]
line_y = [min_val, max_val]

# Add the y=x line to the scatter plot
fig.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='y=x'))


# Set x-axis and y-axis labels
fig.update_layout(
    xaxis_title="Normal Logit Difference",
    yaxis_title="Logit Difference When Freezing Just MLP"
)
# %%
