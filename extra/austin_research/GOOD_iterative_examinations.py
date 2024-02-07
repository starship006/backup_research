"""
This code tests the following prediction:

**Prediction 1: If I ablate head(s), and take all the downstream heads which perform 'self repair' (increase logit contribution), 
they will *still* do it if i copy the resid stream to approximate amounts.**
"""
# %%
import sys

from imports import *
from updated_nmh_dataset_gen import generate_ioi_prompts
from GOOD_helpers import collect_direct_effect
from reused_hooks import overwrite_activation_hook, add_and_replace_hook, store_item_in_tensor

%load_ext autoreload
%autoreload 2
from GOOD_helpers import *
 

# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    print("Running in notebook mode")
    #%load_ext autoreload
    #%autoreload 2
    model_name = "pythia-410m"
else:
    print("Running in script mode")
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
batch_size = 50 # 20 -- if model large    -- 40 else
prompt_len = 40 # 30 -- if model large    -- 30 else

owt_dataset = utils.get_dataset("owt")
owt_dataset_name = "owt"    
    
all_owt_tokens = model.to_tokens(owt_dataset[0:(batch_size * 2)]["text"]).to(device)
owt_tokens = all_owt_tokens[0:batch_size][:, :prompt_len]
corrupted_owt_tokens = all_owt_tokens[batch_size:batch_size * 2][:, :prompt_len]
assert owt_tokens.shape == corrupted_owt_tokens.shape == (batch_size, prompt_len)

# %%
clean_tokens: Float[Tensor, "batch pos"] = owt_tokens
# %%
logits, cache = model.run_with_cache(clean_tokens)
# %% First, see which heads in the model are even useful for predicting these (across ALL positions_)
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)
if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")

correct_logits: Float[Tensor, "batch pos_minus_one"] = get_correct_logit_score(logits, owt_tokens)

# %%
head_to_ablate = (22,11)
model.reset_hooks()
new_cache = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = True)
model.reset_hooks()
new_logits = act_patch(model, owt_tokens, [Node("z", i, j) for (i,j) in [head_to_ablate]], return_item, corrupted_owt_tokens, apply_metric_to_cache = False)
ablated_correct_logit = get_correct_logit_score(new_logits, owt_tokens)

FREEZE_FINAL_LN = False
if FREEZE_FINAL_LN:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False, cache_for_scaling = cache)
else:
    ablated_per_head_direct_effect, ablated_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=owt_tokens, model = model, display = False, collect_individual_neurons = False)



all_head_diff: Float[Tensor, "layer head batch pos"] = (ablated_per_head_direct_effect - per_head_direct_effect)
all_mlp_diff: Float[Tensor, "layer batch pos"] = (ablated_all_layer_direct_effect - all_layer_direct_effect)

# %%
diff_in_logits = (ablated_correct_logit - correct_logits)

#histogram((diff_in_logits).flatten(), title = "Change in Correct Logit Score")
scatter(per_head_direct_effect[head_to_ablate].flatten(), diff_in_logits.flatten(), title = "DE of head vs Change in Correct Logit Score")

self_repair_displayed = diff_in_logits + per_head_direct_effect[head_to_ablate]
scatter(per_head_direct_effect[head_to_ablate].flatten(), self_repair_displayed.flatten(), title = "DE of head vs Self Repair (measured in logits)")


# %%
def simulate_attention_layer_with_different_residual_stream(layer: int, residual_stream: Tensor):
    """
    Simulates the attention layer with a different residual stream
    """
    model.reset_hooks()
    
    # replace the current residual stream with the new one
    model.add_hook(utils.get_act_name("resid_pre", layer), partial(overwrite_activation_hook, what_to_overwrite_with=residual_stream))  
    
    output_of_attn_layer = torch.zeros(cache[utils.get_act_name("attn_out", layer)].shape).to(device)
    model.add_hook(utils.get_act_name("attn_out", layer), partial(store_item_in_tensor, tensor = output_of_attn_layer))
    
    # replace next residual stream, accounting for new attention layer
    if model.cfg.parallel_attn_mlp:
        next_activation = utils.get_act_name("resid_post", layer)
    else:
        next_activation = utils.get_act_name("resid_mid", layer)
        
    updated_resid = cache[next_activation] - cache[utils.get_act_name("attn_out", layer)] 
    model.add_hook(next_activation, partial(add_and_replace_hook, what_to_overwrite_with = updated_resid, added_component_in_list = [output_of_attn_layer])) 
    
    # run the model
    new_logits, new_cache = model.run_with_cache(clean_tokens)
    model.reset_hooks()
    return new_logits, new_cache
    
    
# %%
layer_to_replace = head_to_ablate[0] + 1

if layer_to_replace >= model.cfg.n_layers:
    raise ValueError("welp seems like the last layer is the one to emphasize this, ignore all of this then LOL")


# %%
moved_logits, moved_cache = simulate_attention_layer_with_different_residual_stream(layer_to_replace, cache[utils.get_act_name("resid_pre", layer_to_replace - 1)])

if FREEZE_FINAL_LN:
    moved_per_head_direct_effect, moved_all_layer_direct_effect = collect_direct_effect(moved_cache, correct_tokens=clean_tokens,model = model,                                                                        display=in_notebook_mode, cache_for_scaling=cache)
else:
    moved_per_head_direct_effect, moved_all_layer_direct_effect = collect_direct_effect(moved_cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)
# %%
if True:
    head_diff: Float[Tensor, "layer head batch pos_minus_one"] = moved_per_head_direct_effect - per_head_direct_effect
    layer_diff = moved_all_layer_direct_effect - all_layer_direct_effect
else:
    head_diff = all_head_diff
    layer_diff = all_mlp_diff

if in_notebook_mode:
    show_input(head_diff.mean((-1, -2)),
                layer_diff.mean((-1, -2)), title = "Change in Direct Effect of Heads and MLP Layers")
    imshow(head_diff.mean((-1, -2)))
# %%
print("Average direct effect", per_head_direct_effect[head_to_ablate].mean((-1, -2)))

DE_diff_in_heads = head_diff.sum((-1, -2))
print("Mean DE diff in heads", DE_diff_in_heads.mean((-1, -2)))

#print("Average self repair when ablating", head_diff[layer_to_replace, :].mean((-1, -2)).sum(-1))


# %%
#histogram(head_diff[layer_to_replace, 8].flatten())
histogram(head_diff[layer_to_replace].sum(0).flatten())
# %%
scatter(per_head_direct_effect[head_to_ablate].flatten(), head_diff[layer_to_replace].sum(0).flatten(), title = f"instances of self-repair but when simulating resid layer here")
# %%

for i in range(model.cfg.n_heads):
    scatter(per_head_direct_effect[head_to_ablate].flatten(), head_diff[layer_to_replace, i].flatten(), title = f"Head {i} Direct Effect vs Change in Direct Effect when copying old resid")


# %% 
""" 
Second Approach: Instead of looking at the change in logits, look at cossim with the original head 

Idea: in the past, we've mostly restricted ourselves to look at self-repair in terms of 'does anything recover the output on logits', as though this was the end goal.
But there are many other functions that attention heads could be performing. We can capture that within the notion of looking at 
cossim between the outputs of attention heads to see if they are 'similar' to each other, which would indicate that they possibly
have similar functions.
"""



def get_cossims_from_cache(clean_direction, cache, ignore_token_prediction = False, tokens = None):
    """
    given a clean direction, get the cossims of all heads and layers with that direction
    
    if ignore_token_prediction, we ignore the last token (as we don't know the next prediction to predict away from)
        - i wrote this code out but its kinda useless since model isn't just predicting next token
    """
    
    assert clean_direction.shape == cache[utils.get_act_name("result", 0)][..., 0, :].shape
    assert clean_direction.shape == cache[utils.get_act_name("mlp_out", 0)].shape
    
    
    if tokens is not None:
        token_dirs = model.tokens_to_residual_directions(tokens[:, 1:])
    else:
        token_dirs = None
        
    def project_away(clean_direction):
        assert token_dirs is not None
        assert ignore_token_prediction
        #print(clean_direction.shape, token_dirs.shape)
        projection = get_3d_projection(clean_direction, token_dirs)
        return clean_direction - projection
    
    
    if ignore_token_prediction:
        clean_direction = clean_direction[:, :-1, :]
        clean_direction = project_away(clean_direction)
    
    
    cossim_prompt_len = prompt_len - 1 if ignore_token_prediction else prompt_len
    head_cossims = torch.zeros((model.cfg.n_layers, model.cfg.n_heads, batch_size, cossim_prompt_len))
    layer_cossims = torch.zeros((model.cfg.n_layers, batch_size, cossim_prompt_len))
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            head_out = cache[utils.get_act_name("result", layer)][..., head, :] 
            if ignore_token_prediction:
                head_out = head_out[:, :-1, :]
                head_out = project_away(head_out)
            head_cossims[layer, head] = F.cosine_similarity(clean_direction, head_out, dim = -1)
    
    
        layer_out = cache[utils.get_act_name("mlp_out", layer)]
        
        if ignore_token_prediction:
            layer_out = layer_out[:, :-1, :]
            layer_out = project_away(layer_out)
            
        layer_cossims[layer] = F.cosine_similarity(clean_direction, layer_out, dim = -1)
    return head_cossims, layer_cossims


# %% Finding heads which seem to have large cossim with other heads around it (indication of model iterativity?)
max_cossim_with_other_head = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        clean_direction: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", layer)][..., head, :]
        
        head_cossims, _ = get_cossims_from_cache(clean_direction, cache, ignore_token_prediction = False, tokens = clean_tokens)
        head_cossims = head_cossims.mean((-1, -2))
        head_cossims[layer, head] = 0
        max_cossim_with_other_head[layer, head] = head_cossims.abs().max()

        
# %%
imshow(max_cossim_with_other_head, title = "Max Cosine Similarity of Heads with Other Heads")

# %%
random_head = (16, 15)
clean_direction: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", random_head[0])][..., random_head[1], :]
head_cossims, layer_cossims = get_cossims_from_cache(clean_direction, cache, ignore_token_prediction = False, tokens = clean_tokens)

# %%
show_input(head_cossims.mean((-1, -2)), layer_cossims.mean((-1, -2)), title = f"Average Cosine Similarity of Heads and MLP Layers w/ {random_head}")

# %%
nodes = [Node("z", i, j) for (i,j) in [random_head]]
ablated_cache = act_patch(model, clean_tokens, nodes, return_item, corrupted_owt_tokens, apply_metric_to_cache=True)
ablated_logits = act_patch(model, clean_tokens, nodes, return_item, corrupted_owt_tokens, apply_metric_to_cache=False)
ablated_correct_logits = get_correct_logit_score(ablated_logits, clean_tokens)
# %%
new_head_cossims, new_layer_cossims = get_cossims_from_cache(clean_direction, ablated_cache)

# %%
diff_head = new_head_cossims - head_cossims
diff_mlp = new_layer_cossims - layer_cossims

diff_head[random_head[0], random_head[1]] = 0 # just for visual purposes
#imshow(diff.mean((-1, -2)), title = "Change in Average Cosine Similarity of Heads with Original Head")
show_input(diff_head.mean((-1, -2)), diff_mlp.mean((-1, -2)), title = "Change in Average Cosine Similarity of Heads and MLP Layers")
# %%
top_instances = topk_of_Nd_tensor(correct_logits - ablated_correct_logits, 10)
# %%
instance = 1
batch = top_instances[instance][0]
pos = top_instances[instance][1]
print("Change in logits", correct_logits[batch, pos] - ablated_correct_logits[batch, pos])
#imshow(clean_cossims[..., batch, pos], title = f"Average Cosine Similarity of Heads with Original Head on B{batch}P{pos}")
show_input(head_cossims[..., batch, pos], layer_cossims[..., batch, pos], title = f"Average Cosine Similarity of Heads and MLP Layers with Original Head on B{batch}P{pos}")
#imshow(diff[..., batch, pos], title = f"Change in Average Cosine Similarity of Heads with Original Head on B{batch}P{pos}")
show_input(diff_head[..., batch, pos], diff_mlp[..., batch, pos], title = f"Change in Average Cosine Similarity of Heads and MLP Layers with Original Head on B{batch}P{pos}")
# %%
imshow(per_head_direct_effect[..., batch, pos], title = f"Direct Effect of Heads on B{batch}P{pos}")

# %% 
interesting_head = (17, 7)
def head_de_vs_head_de(head_one: tuple, head_two: tuple):
    x = per_head_direct_effect[head_one].flatten()
    y = per_head_direct_effect[head_two].flatten()
    scatter(x, y, title = f"Direct Effect of Head {head_one} vs Head {head_two}")
    
    
head_de_vs_head_de(random_head, interesting_head)


# %%

def head_vs_head_cossims(head_from: tuple, head_to: tuple):
    clean_direction: Float[Tensor, "batch pos d_model"] = cache[utils.get_act_name("result", head_from[0])][..., head_from[1], :]    
    head_cossims, _ = get_cossims_from_cache(clean_direction, cache, ignore_token_prediction = False, tokens = clean_tokens)
    
    x = per_head_direct_effect[head_from[0], head_from[1]].flatten()
    y = head_cossims[head_to[0], head_to[1], :, :-1].flatten() 
    
    
    
    # Scatter plot in plotly
    fig = px.scatter(x=x.cpu().numpy(), y=y.cpu().numpy(), title = f"Direct Effect of Head {head_from} vs Cosine Similarity of Head {head_to} with Head {head_from}")
    # decrease dot size
    fig.update_traces(marker=dict(size=2))
    # relabel axes
    fig.update_layout(
        xaxis_title = f"Direct Effect of Head {head_from}",
        yaxis_title = f"Cossim between {head_to} and {head_from}"
    )
    fig.show()
    
    hist(y, title = f"Cosine Similarity of Head {head_to} with Head {head_from}")
    

head_vs_head_cossims(random_head, interesting_head)
    
    

# %%
