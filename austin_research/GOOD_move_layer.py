"""
This code tests the hypothesis that moving the residual stream from layer NMH layer to downstream layer causes one layer down to more name moving.
"""
# %%
from imports import *
from updated_nmh_dataset_gen import generate_ioi_prompts
from GOOD_helpers import collect_direct_effect
from reused_hooks import overwrite_activation_hook
# %%
%load_ext autoreload
%autoreload 2
from GOOD_helpers import *
# %% Constants
in_notebook_mode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "pythia-1b-deduped"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name


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
PROMPTS, ANSWERS, ANSWER_INDICIES = generate_ioi_prompts(model, 30)
# %%
clean_tokens: Float[Tensor, "batch pos"] = model.to_tokens(PROMPTS).to(device)
answer_tokens: Float[Tensor, "batch 1"] = model.to_tokens(ANSWERS, prepend_bos=False).to(device)
answer_token_idx: Float[Tensor, "batch"] = torch.tensor(ANSWER_INDICIES).to(device)

# %%
# See if the answer_token_idx is correct
unsqueezed_answers_idx = answer_token_idx.unsqueeze(-1)
indexed_answers = clean_tokens.gather(-1, unsqueezed_answers_idx)
assert torch.all(torch.eq(answer_tokens, indexed_answers))

# %%
logits, cache = model.run_with_cache(clean_tokens)
# %%
# %% First, see which heads in the model are even useful for predicting these
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)

if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")
# %% Get the prediction of the names

def get_name_direct_effects(per_head_direct_effect, all_layer_direct_effect, answer_token_idx):
    expanded_indicies = einops.repeat(answer_token_idx, "b -> a c b 1", a = model.cfg.n_layers, c = model.cfg.n_heads)
    important_direct_effect: Float[Tensor, "layer head batch"] = per_head_direct_effect.gather(-1, expanded_indicies - 1).squeeze()

    expanded_indicies_mlp = einops.repeat(answer_token_idx, "b -> a b 1", a = model.cfg.n_layers,)
    important_direct_effect_mlp = all_layer_direct_effect.gather(-1, expanded_indicies_mlp - 1).squeeze()
    return important_direct_effect, important_direct_effect_mlp


important_direct_effect, important_direct_effect_mlp = get_name_direct_effects(per_head_direct_effect, all_layer_direct_effect, answer_token_idx)
# %%
if in_notebook_mode:
    show_input(important_direct_effect.mean((-1)), important_direct_effect_mlp.mean((-1)), title = "Direct Effect of Heads and MLP Layers on predicting Name")

# %%
def run_forward_pass_and_copy_layer(from_activation: str, to_activation: str):
    """
    does a forwawrd pass but copies the residuals tream from one activation to another
    """

    from_activation = cache[from_activation]
    model.reset_hooks()
    model.add_hook(to_activation, partial(overwrite_activation_hook, what_to_overwrite_with=from_activation))
    
    new_logits, new_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    return new_logits, new_cache
# %% Have the from layer be the layer with the most direct effect
from_layer = important_direct_effect.sum((-1,-2)).argmax().item()
to_layer = from_layer + 1

from_act = utils.get_act_name("resid_pre", from_layer)
to_act = utils.get_act_name("resid_pre", to_layer)

if to_layer >= model.cfg.n_layers:
    print("welp seems like the last layer is the one to emphasize this, ignore all of this then LOL")
else:
    print(f"We are ablating from {from_layer} to {to_layer}")


# %%
moved_logits, moved_cache = run_forward_pass_and_copy_layer(from_act, to_act)
# %%
moved_per_head_direct_effect, moved_all_layer_direct_effect = collect_direct_effect(moved_cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode, cache_for_scaling=cache)
moved_important_direct_effect, moved_important_direct_effect_mlp = get_name_direct_effects(moved_per_head_direct_effect, moved_all_layer_direct_effect, answer_token_idx)
# %%
if in_notebook_mode:
    show_input(moved_important_direct_effect.mean((-1)) - important_direct_effect.mean((-1)),
                moved_important_direct_effect_mlp.mean((-1)) - important_direct_effect_mlp.mean((-1)), title = f"Change in DE when copying from {from_act} to {to_act}")
# %%
move_layer_diff = moved_important_direct_effect.mean((-1)) - important_direct_effect.mean((-1))
print(move_layer_diff.sum(-1))
print(move_layer_diff.sum(-1)[to_layer])


important_layer_change_from_move_layer = move_layer_diff[to_layer]

# %%

# %% Ablate the normal head with something random

# %%
owt_dataset = utils.get_dataset("owt")
owt_dataset_name = "owt"
# %%
batch_size = clean_tokens.shape[0]
prompt_len = clean_tokens.shape[1]

all_owt_tokens = model.to_tokens(owt_dataset[0:(batch_size * 2)]["text"]).to(device)
owt_tokens = all_owt_tokens[0:batch_size][:, :prompt_len]
corrupted_owt_tokens = all_owt_tokens[batch_size:batch_size * 2][:, :prompt_len]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (batch_size, prompt_len)
# %%
top_head = topk_of_Nd_tensor(important_direct_effect.mean((-1)), 1)[0]

assert top_head[0] == from_layer

nodes = [Node("z", top_head[0], top_head[1])]
new_cache = act_patch(model, clean_tokens, nodes, return_item, owt_tokens, apply_metric_to_cache=True)
ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=clean_tokens,model = model,
                                                                                        title = "ablated DE", display=in_notebook_mode, cache_for_scaling=cache)
ablated_important_direct_effect, ablated_important_direct_effect_mlp = get_name_direct_effects(ablated_per_head_direct_effect, ablated_all_layer_direct_effect, answer_token_idx)
# %%
if in_notebook_mode:
    show_input(ablated_important_direct_effect.mean((-1)) - important_direct_effect.mean((-1)),
                ablated_important_direct_effect_mlp.mean((-1)) - important_direct_effect_mlp.mean((-1)), title = f"Change in DE when ablating head")
# %%
ablation_diff = ablated_important_direct_effect.mean((-1)) - important_direct_effect.mean((-1))
print(ablation_diff.sum(-1))
print(ablation_diff.sum(-1)[to_layer])


change_from_ablation = ablation_diff[to_layer]
# %% 

px.scatter(x = important_layer_change_from_move_layer.cpu(), y = change_from_ablation.cpu(), labels = {"x": "Change in DE from moving layer", "y": "Change in DE from ablating head"})
# %%
