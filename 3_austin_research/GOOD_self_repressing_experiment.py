"""
Goals with this research direction:
- Figure out whether or not heads are self-repressing; i.e., whether or not reading in more of their output causes them to do less of a task.
"""
# %%
import sys
from enum import Enum
# %%
from imports import *
from GOOD_helpers import *
from reused_hooks import overwrite_activation_hook
# %%
#%load_ext autoreload
#%autoreload 2

# %% Constants
in_notebook_mode = False
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
batch_size = 30
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
top_heads = topk_of_Nd_tensor(per_head_direct_effect.mean((-1,-2)), 2)

global_top_head = top_heads[0]
global_second_head = top_heads[1]


# %%
class output_source(Enum):
    WITH_SAME_AS_RECEIVING_HEAD = 1
    WITH_RANDOM_HEAD = 2
    WITH_RANDOM_DIRECTION = 3
    WITH_FIXED_GLOBAL_TOP_HEAD = 4

class receiving_source(Enum):
    WITH_TOP_HEAD = 1
    WITH_FIXED_GLOBAL_TOP_HEAD = 2
    WITH_CUSTOM_HEAD = 3
    WITH_FIXED_GLOBAL_SECOND_HEAD = 4


def analyze_head(output_type, receiving_type, scaling, custom_head = None):
    original_de = torch.zeros((batch_size, prompt_len - 1))
    new_de = torch.zeros((batch_size, prompt_len - 1))
    receive_heads = []
    
    for batch in range(batch_size):
        for pos in range(prompt_len - 1):
            direct_effects = per_head_direct_effect[..., batch, pos]
            
            if receiving_type == receiving_source.WITH_FIXED_GLOBAL_TOP_HEAD:
                receiving_head = global_top_head
            elif receiving_type == receiving_source.WITH_FIXED_GLOBAL_SECOND_HEAD:
                receiving_head = global_second_head
            elif receiving_type == receiving_source.WITH_TOP_HEAD:
                receiving_head = topk_of_Nd_tensor(direct_effects, 1)[0]
            elif receiving_type == receiving_source.WITH_CUSTOM_HEAD:
                if custom_head is None:
                    raise Exception("Must provide custom_head if receiving_type is WITH_CUSTOM_HEAD")
                receiving_head = custom_head
            else:
                raise Exception("Invalid receiving_type")
            
            # test what happens when you take its output and feed it back into itself
            if output_type == output_source.WITH_RANDOM_HEAD:
                output_head = (torch.randint(0, model.cfg.n_layers, (1,)).item(), torch.randint(0, model.cfg.n_heads, (1,)).item())
                head_out: Float[Tensor, "d_model"] = cache[utils.get_act_name("result", output_head[0])][batch, pos, output_head[1], :]
            elif output_type == output_source.WITH_SAME_AS_RECEIVING_HEAD:
                output_head = receiving_head
                head_out: Float[Tensor, "d_model"] = cache[utils.get_act_name("result", output_head[0])][batch, pos, output_head[1], :]
            elif output_type == output_source.WITH_RANDOM_DIRECTION:
                temp_output_head = receiving_head
                temp_head_out: Float[Tensor, "d_model"] = cache[utils.get_act_name("result", temp_output_head[0])][batch, pos, temp_output_head[1], :]
                rand_dir =  torch.randn(model.cfg.d_model).to(device)
                head_out: Float[Tensor, "d_model"] = (rand_dir / rand_dir.norm()) * temp_head_out.norm()
            elif output_type == output_source.WITH_FIXED_GLOBAL_TOP_HEAD:
                head_out: Float[Tensor, "d_model"] = cache[utils.get_act_name("result", global_top_head[0])][batch, pos, global_top_head[1], :]
            else:
                raise Exception("Invalid output_type")
            
            
            repeated_head_out = einops.repeat(head_out, "d_model -> batch d_model", batch = batch_size).to(device)
            model.reset_hooks()
            model.add_hook(utils.get_act_name("resid_pre", receiving_head[0]), partial(add_vector_to_resid, vector = repeated_head_out * scaling, positions = pos))
            new_logits, new_cache = model.run_with_cache(clean_tokens)
            model.reset_hooks()
            
            
            # see if the direct effect of the head has changed
            new_per_head_direct_effect, new_all_layer_direct_effect = collect_direct_effect(new_cache, correct_tokens=clean_tokens,
                                                                                model = model, display=False)
            
            original_de[batch, pos] = (per_head_direct_effect)[receiving_head[0], receiving_head[1], batch, pos].item()
            new_de[batch, pos] = (new_per_head_direct_effect)[receiving_head[0], receiving_head[1], batch, pos].item()
            receive_heads.append(receiving_head)
            
    return original_de, new_de, receive_heads
# %
# %%
scaling_factors = [1, 2, 5]
def loop_and_analyze():
    results = {}
    
    # try to load in results
    try:
        with open(f'results_{safe_model_name}.pickle', 'rb') as f:
            results = pickle.load(f)
    except:
        pass
    
    for output_type in tqdm([output_source.WITH_SAME_AS_RECEIVING_HEAD, output_source.WITH_RANDOM_HEAD, output_source.WITH_RANDOM_DIRECTION]):
        for receiving_type in [receiving_source.WITH_TOP_HEAD, receiving_source.WITH_FIXED_GLOBAL_TOP_HEAD, receiving_source.WITH_FIXED_GLOBAL_SECOND_HEAD]:
            for scaling in scaling_factors:
                if (output_type.name, receiving_type.name, scaling) in results:
                    print("already did this comp")
                    continue
                
                print("starting new comp")
                orig_de, new_de, heads = analyze_head(output_type, receiving_type, scaling = scaling)
                key = (output_type.name, receiving_type.name, scaling)
                results[key] = (orig_de, new_de, heads)
                #plot_results(orig_de, new_de, heads, scaling, output_type, receiving_type)
                
                
    with open(f'results_{safe_model_name}.pickle', 'wb') as f:
        pickle.dump(results, f)

loop_and_analyze()


# %%


# %%
