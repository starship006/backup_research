"""
Goals with this research direction:
- Figure out whether or not heads are self-repressing; i.e., whether or not reading in more of their output causes them to do less of a task.
"""
# %%
import sys
# %%
from imports import *
from GOOD_helpers import *
from reused_hooks import overwrite_activation_hook
# %%
%load_ext autoreload
%autoreload 2

# %% Constants
in_notebook_mode = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "gpt2-small"
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
batch_size = 20
prompt_len = 30

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
# %%
# %% First, see which heads in the model are even useful for predicting these (across ALL positions_)
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=in_notebook_mode)

if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")

# %%
SCALING = 1
OUTPUT_WITH_RANDOM_HEAD = False
OUTPUT_WITH_RANDOM_DIRECTION = True


original_de = torch.zeros((batch_size, prompt_len - 1))
new_de = torch.zeros((batch_size, prompt_len - 1))
for batch in range(batch_size):
    for pos in range(prompt_len - 1):
        direct_effects = per_head_direct_effect[..., batch, pos]
        top_head = topk_of_Nd_tensor(direct_effects, 1)[0]
        

        # test what happens when you take its output and feed it back into itself
        if OUTPUT_WITH_RANDOM_HEAD:
            output_head = (torch.randint(0, model.cfg.n_layers, (1,)).item(), torch.randint(0, model.cfg.n_heads, (1,)).item())
        else:
            output_head = top_head
        head_out: Float[Tensor, "d_model"] = cache[utils.get_act_name("result", output_head[0])][batch, pos, output_head[1], :]
        if OUTPUT_WITH_RANDOM_DIRECTION:
            rand_dir =  torch.randn(model.cfg.d_model).to(device)
            head_out = (rand_dir / rand_dir.norm()) * head_out.norm()
        
        repeated_head_out = einops.repeat(head_out, "d_model -> batch d_model", batch = batch_size).to(device)
        
        model.reset_hooks()
        model.add_hook(utils.get_act_name("resid_pre", top_head[0]), partial(add_vector_to_resid, vector = repeated_head_out * SCALING, positions = pos))
        new_logits, new_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()
        
        
        # see if the direct effect of the head has changed
        new_per_head_direct_effect, new_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=clean_tokens,
                                                                            model = model, display=False)
        
        #print(f"Diff in DE of {top_head} = " + str((new_per_head_direct_effect - per_head_direct_effect)[top_head[0], top_head[1], batch, pos].item()) + "| OG = " + str(per_head_direct_effect[top_head[0], top_head[1], batch, pos].item()))
        original_de[batch, pos] = (per_head_direct_effect)[top_head[0], top_head[1], batch, pos].item()
        new_de[batch, pos] = (new_per_head_direct_effect)[top_head[0], top_head[1], batch, pos].item()
    
# %%
if OUTPUT_WITH_RANDOM_HEAD:
    prefix = "random-head" 
elif OUTPUT_WITH_RANDOM_DIRECTION:
    prefix = "random-direction"
else:
    prefix = "self"
fig = px.scatter(x=original_de.flatten(), y=new_de.flatten(), title=f"{prefix}-repression of heads | scaling == {SCALING}")

fig.add_trace(go.Scatter(x=[original_de.min(), original_de.max()], y=[original_de.min(), original_de.max()], mode='lines', name='y=x'))
fig.update_layout(
    xaxis_title="Original Direct Effect",
    yaxis_title="New Direct Effect",
)

fig.update_traces(marker=dict(size=2))  # Decrease the size of the dots

fig.show()

# %%
