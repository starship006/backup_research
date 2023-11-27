"""
Goals with this research direction:
- Flesh out the hypothesis that model's are iterative, and test to what extent this is true
- If this is indeed the case, find a way to tie it back to backup and self-repair
"""
# %%
from imports import *
from updated_nmh_dataset_gen import generate_singular_ioi_prompt_type, generate_ioi_prompts
from GOOD_helpers import *
from reused_hooks import overwrite_activation_hook
# %%
%load_ext autoreload
%autoreload 2

# %% Constants
in_notebook_mode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "pythia-160m"
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
PROMPTS, ANSWERS, ANSWER_INDICIES = generate_singular_ioi_prompt_type(model, 30)
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
# %% First, see which heads in the model are even useful for predicting these (across ALL positions_)
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)

if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")
# %% Check for name prediction
def get_name_direct_effects(per_head_direct_effect, all_layer_direct_effect, answer_token_idx):
    expanded_indicies = einops.repeat(answer_token_idx, "b -> a c b 1", a = model.cfg.n_layers, c = model.cfg.n_heads)
    important_direct_effect: Float[Tensor, "layer head batch"] = per_head_direct_effect.gather(-1, expanded_indicies - 1).squeeze()

    expanded_indicies_mlp = einops.repeat(answer_token_idx, "b -> a b 1", a = model.cfg.n_layers,)
    important_direct_effect_mlp = all_layer_direct_effect.gather(-1, expanded_indicies_mlp - 1).squeeze()
    return important_direct_effect, important_direct_effect_mlp


important_direct_effect, important_direct_effect_mlp = get_name_direct_effects(per_head_direct_effect, all_layer_direct_effect, answer_token_idx)

if in_notebook_mode:
    show_input(important_direct_effect.mean((-1)), important_direct_effect_mlp.mean((-1)), title = "Direct Effect of Heads and MLP Layers on predicting Name")

# %% Isolate the output of a single head
top_heads = topk_of_Nd_tensor(important_direct_effect.mean((-1)), 5)
top_head: tuple = (top_heads[0][0], top_heads[0][1])
second_head: tuple = (top_heads[1][0], top_heads[1][1])




def run_forward_pass_and_copy_layer(resid_layer_to_add: int, vector: Float[Tensor, "batch d_model"], scaling = 1):
    """
    does a forwawrd pass but adds scaled version of a vector to at resid_pre_{resid_layer_to_add} multiplied by some sort of scaling
    """

    
    model.reset_hooks()
    model.add_hook(utils.get_act_name("resid_pre", resid_layer_to_add), partial(add_vector_to_resid, vector = vector * scaling, positions = answer_token_idx - 1))
    
    new_logits, new_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    return new_logits, new_cache
# %%

def add_vector_resid_see_change(head: tuple, scaling = 1, layer = None, display_change = True):
    """
    given a head, adds the output vectors of the head on the positions predicting the name, and then outputs the difference in the direct effect of the heads and the mlp layers
    
    if layer is None, then it will add the vector to the next layer
    """
    
    if layer is None:
        layer = head[0] + 1
    
    head_out: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", head[0])][..., head[1], :]
    expanded_pre_answer_indicies = einops.repeat(answer_token_idx - 1, "batch -> batch 1 d_model", d_model = model.cfg.d_model)
    target_head_out: Float[Tensor, "batch d_model"] = head_out.gather(-2, expanded_pre_answer_indicies).squeeze()


    ablated_logits, ablated_cache = run_forward_pass_and_copy_layer(layer, target_head_out, scaling = scaling)
    ablated_important_direct_effect, ablated_important_direct_effect_mlp = get_name_direct_effects(*collect_direct_effect(ablated_cache, correct_tokens=clean_tokens,model = model,
                                                                        display=False, cache_for_scaling=cache), answer_token_idx)

    if display_change:
        head_de = ablated_important_direct_effect.mean((-1)) - important_direct_effect.mean((-1))
        mlp_de = ablated_important_direct_effect_mlp.mean((-1)) -  important_direct_effect_mlp.mean((-1))
        title = f"Difference in Direct Effect of Heads and MLP Layers on predicting Name when adding scaling={scaling} output of head {head}"
    else:
        head_de = ablated_important_direct_effect.mean((-1))
        mlp_de = ablated_important_direct_effect_mlp.mean((-1))
        title = f"Direct Effect of Heads and MLP Layers on predicting Name when adding scaling={scaling} output of head {head}"
    
        
    show_input(head_de, mlp_de, title = title)

add_vector_resid_see_change(second_head, -3, second_head[0], True)
# %%
