# %%
from imports import *
import argparse
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score

# %%
in_notebook_mode = True
if in_notebook_mode:
    model_name = "pythia-160m"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name

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
#BATCH_SIZE = 300
#PROMPT_LEN = 50
BATCH_SIZE = 150
PROMPT_LEN = 30


if owt_dataset_name == "owt":
    bad_prompts =  [149] # outlier examples; maybe red team some day? for the 
    #bad_prompts =  [200, 174, 243, 149] # outlier examples; maybe red team some day? for the 
else:
    bad_prompts = []

# %% Given above batch/prompt descriptions, load cached data
SAMPLE_ABLATE_CORRECT_LOGIT_DIR_NAME = f"pickle_storage/{safe_model_name}_{owt_dataset_name}_sample_ablate_new_logits_cache_{BATCH_SIZE}_{PROMPT_LEN}.pkl"
new_logits_after_sample = {}
try:
    with open(SAMPLE_ABLATE_CORRECT_LOGIT_DIR_NAME, 'rb') as f:
        new_logits_after_sample = pickle.load(f)
except FileNotFoundError:
    print("oh, new combo? guess we gonna have to rebuild the cache")
    

# %% Load tokens
all_owt_tokens = model.to_tokens(owt_dataset[0:(BATCH_SIZE * 2 + len(bad_prompts))]["text"]).to(device)
owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
extra_prompts = all_owt_tokens[(BATCH_SIZE * 2):][:, :PROMPT_LEN]


for i, prompt in enumerate(bad_prompts):
  owt_tokens[prompt] = extra_prompts[i]



assert owt_tokens.shape == corrupted_owt_tokens.shape == (BATCH_SIZE, PROMPT_LEN)

# %%
logits, cache = model.run_with_cache(owt_tokens)

print(utils.lm_accuracy(logits, owt_tokens))
print(utils.lm_cross_entropy_loss(logits, owt_tokens))
# %% Import the helper functions from the helpers.py file and pass in shared arguments

partials = return_partial_functions(model = model, clean_tokens = owt_tokens, corrupted_tokens = corrupted_owt_tokens, cache = cache)
globals().update(partials)

# %%
# get per component direct effects
# this takes TONS of memory, i should decrease
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=owt_tokens, display = in_notebook_mode, collect_individual_neurons = False)


# %%
show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)))
#show_input(per_head_direct_effect.mean((-1,-2)),torch.zeros((model.cfg.n_layers)))

# # %%
# histogram(per_head_direct_effect[9,5].flatten(), title = "direct effects of L9H5") # this is a copy suppression head in the model maybe?
# # %%
# histogram(all_layer_direct_effect[-1].flatten())

# %%
def print_tokens(batch, start = 40, end = 47):
    """
    Prints the tokens for a batch. Shares same indexing.
    """
    print("Tokens before:")
    print(model.to_string(all_owt_tokens[batch, 0:start]))
    print("Start token == " , model.to_string(all_owt_tokens[batch, start]))
    print("Tokens after:")
    print(model.to_string(all_owt_tokens[batch, start + 1:end]))
    # print("...")
    # print(model.to_string(all_owt_tokens[batch, end:]))



def change_in_ld_calc(heads = None, mlp_layers = None, num_runs = 5, logits = logits):
    """
    runs activation patching over component and returns new avg_correct_logit_score, averaged over num_runs runs
    this is the average logit of the correct token upon num_runs sample ablations
    logits just used for size
    """
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    # use many shuffled!
    logits_accumulator = torch.zeros_like(logits)
    for i in range(num_runs):
        # Shuffle owt_tokens by batch
        shuffled_corrupted_owt_tokens = shuffle_owt_tokens_by_batch(corrupted_owt_tokens)
        # Calculate new_logits using act_patch
        new_logits = act_patch(model, owt_tokens, nodes, return_item, shuffled_corrupted_owt_tokens, apply_metric_to_cache=False)
        logits_accumulator += new_logits

    avg_logits = logits_accumulator / num_runs
    # get change in direct effect compared to original
    avg_correct_logit_score = get_correct_logit_score(avg_logits)
    return avg_correct_logit_score

def new_logits_after_sample_wrapper(heads=None, mlp_layers=None, num_runs=5, logits=logits):
    if num_runs != 5:
        raise Exception("num_runs must be 5 for now")
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        direct_effect = per_head_direct_effect[layer, head]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
        direct_effect = all_layer_direct_effect[layer]
    else:
        raise Exception("No heads or mlp layers given")
    

    key = "H:" + str(heads) +  "M:" + str(mlp_layers)
    if key in new_logits_after_sample:
        print("Returning cached activation")
        return new_logits_after_sample[key]
    
    result = change_in_ld_calc(heads, mlp_layers, num_runs, logits)
    
    new_logits_after_sample[key] = result
    
    with open(SAMPLE_ABLATE_CORRECT_LOGIT_DIR_NAME, 'wb') as f:
        pickle.dump(new_logits_after_sample, f)
        
    return result

def show_batch_result(batch, start = 40, end = 47, per_head_direct_effect = per_head_direct_effect, all_layer_direct_effect = all_layer_direct_effect):
    """
    highlights the text selection, along with the mean effect of the range
    indexed similariy to python, where start is inclusive and end is exclusive 

    recall that the per_head_direct_effect is one length shorter than the input, since it doesn't have the first token
    so, if the interesting self-repair you are observing seems to be at pos 12, this means it is for the prediction of token 13
    """
    
    print_tokens(start, end)
    show_input(per_head_direct_effect[..., batch, start:end].mean(-1),all_layer_direct_effect[:, batch, start:end].mean(-1), title = f"Direct Effect of Heads on batch {batch}")

def create_scatter_of_backup_of_component(heads = None, mlp_layers = None, return_slope = False, return_CRE = False):
    """"
    this function:
    1) gets the direct effect of all a component when sample ablating it
    2) gets the total accumulated backup of the component for each prompt and position
    3) plots the clean direct effect vs accumulated backup

    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    return_CRE: bad coding practic eto wherei  just return cre if i'm lazy
    """

    # don't accept if more than one input is none
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
    else:
        raise Exception("No heads or mlp layers given")
    
    ablated_per_head_batch_direct_effect, mlp_per_layer_direct_effect = dir_effects_from_sample_ablating(attention_heads=heads, mlp_layers=mlp_layers)

    # 2) gets the total accumulated backup of the head for each prompt and position
    downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
    downstream_change_in_mlp_logit_diff: Float[Tensor, "layer batch pos"] = mlp_per_layer_direct_effect - all_layer_direct_effect
    #print(downstream_change_in_logit_diff.shape)
    #print(downstream_change_in_mlp_logit_diff.shape)
    
    if(downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() != 0):
        print("expect assymetry since different LNs used")
    if heads is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer):].sum(0)
        total_backup = head_backup + mlp_backup
    if mlp_layers is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer+1):].sum(0)
        total_backup = head_backup + mlp_backup
    
    if return_CRE:
        return total_backup
    
    #  3) plots the clean direct effect vs accumulated backup

    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else all_layer_direct_effect[layer].flatten().cpu()
    assert direct_effects.shape == total_backup.flatten().cpu().shape
    if not return_slope:
        fig = go.Figure()
        
        text_labels = [f"Batch {i[0]}, Pos {i[1]}: {model.to_string(all_owt_tokens[i[0], i[1]:(i[1] + 1)])} --> {model.to_string(all_owt_tokens[i[0], (i[1] + 1):(i[1] + 2)])}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]

        scatter_plot = go.Scatter(
            x = direct_effects,
            y = total_backup.flatten().cpu(),
            text=text_labels,  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Compensatory Response Size"
        )
        fig.add_trace(scatter_plot)

        second_scatter = go.Scatter(
            x = direct_effects,
            y = head_backup.flatten().cpu() ,
            text=text_labels,  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Response Size of just Attention Blocks"
        )
        fig.add_trace(second_scatter)

        third_scatter = go.Scatter(
            x = direct_effects,
            y = mlp_backup.flatten().cpu() ,
            text=text_labels, # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Response Size of just MLP Blocks"
        )
        fig.add_trace(third_scatter)


    # get a best fit line
   
    slope, intercept = np.linalg.lstsq(np.vstack([direct_effects, np.ones(len(direct_effects))]).T, total_backup.flatten().cpu(), rcond=None)[0]

    if not return_slope:
        max_x = max(direct_effects)

        # add line of best fit
        fig.add_trace(go.Scatter(
            x=torch.linspace(-1 * max_x,max_x,100),
            y=torch.linspace(-1 * max_x,max_x,100) * slope + intercept,
            mode='lines',
            name='lines'
        ))

        component = heads if heads is not None else mlp_layers
        fig.update_layout(
            title=f"Total Accumulated Backup of {component} in {model_name} for each Position and Batch" if heads is not None 
            else f"Total Accumulated Backup of MLP Layer {component} in {model_name} for each Position and Batch",
        )
        fig.update_xaxes(title = f"Direct Effect of Head {heads[0]}" if heads is not None else f"Direct Effect of MLP Layer {mlp_layers[0]}")
        fig.update_yaxes(title = "Total Accumulated Backup")
        fig.update_layout(width=900, height=500)
        fig.show()
    
    if return_slope:
        return slope
    
def shuffle_owt_tokens_by_batch(owt_tokens):
    batch_size, num_tokens = owt_tokens.shape
    shuffled_owt_tokens = torch.zeros_like(owt_tokens)
    
    for i in range(batch_size):
        perm = torch.randperm(num_tokens)
        shuffled_owt_tokens[i] = owt_tokens[i, perm]
        
    return shuffled_owt_tokens

def create_scatter_of_change_from_component(heads = None, mlp_layers = None, return_slope = False, zero_ablate = False, force_through_origin = False, num_runs = 1, logits = logits):
    """"
    this function:
    1) gets the direct effect of all a component when sample ablating it
    2) gets the CHANGE IN LOGIT CONTRIBUTION for each prompt and position
    3) plots the clean direct effect vs accumulated backup

    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    force_through_origin: if true, will force the line of best fit to go through the origin
    """

    # don't accept if more than one input is none
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
    else:
        raise Exception("No heads or mlp layers given")
    
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    if zero_ablate:
        new_logits = act_patch(model, owt_tokens, nodes, return_item, new_cache="zero", apply_metric_to_cache= False)
        diff_in_logits = new_logits - logits
        change_in_direct_effect = get_correct_logit_score(diff_in_logits)
    else:
        new_logits = new_logits_after_sample_wrapper(heads, mlp_layers, num_runs, logits)
        change_in_direct_effect = new_logits - get_correct_logit_score(logits)
    
    #  3) plots the clean direct effect vs accumulated backup
    direct_effects = per_head_direct_effect[layer, head].flatten().cpu() if heads is not None else all_layer_direct_effect[layer].flatten().cpu()
    assert direct_effects.shape == change_in_direct_effect.flatten().cpu().shape

    # get a best fit line
    if force_through_origin:
        slope = np.linalg.lstsq(direct_effects.reshape(-1, 1), change_in_direct_effect.flatten().cpu(), rcond=None)[0][0]
        intercept = 0
    else:
        slope, intercept = np.linalg.lstsq(np.vstack([direct_effects, np.ones(len(direct_effects))]).T, change_in_direct_effect.flatten().cpu(), rcond=None)[0]
    
    if not return_slope:
        fig = go.Figure()
        text_labels = [f"Batch {i[0]}, Pos {i[1]}: {model.to_string(all_owt_tokens[i[0], i[1]:(i[1] + 1)])} --> {model.to_string(all_owt_tokens[i[0], (i[1] + 1):(i[1] + 2)])}" for i in itertools.product(range(BATCH_SIZE), range(PROMPT_LEN - 1))]


        scatter_plot = go.Scatter(
            x = direct_effects,
            y = change_in_direct_effect.flatten().cpu(),
            text=text_labels,  # Set the hover labels to the text attribute
            mode='markers',
            marker=dict(size=2, opacity=0.8),
            name = "Change in Final Logits vs Direct Effect"
        )
        fig.add_trace(scatter_plot)
        max_x = max(direct_effects.abs())

        # add line of best fit
        fig.add_trace(go.Scatter(
            x=torch.linspace(-1 * max_x,max_x,100),
            y=torch.linspace(-1 * max_x,max_x,100) * slope + intercept,
            mode='lines',
            name='lines'
        ))

        # add dashed y=-x line
        fig.add_trace(go.Scatter(
            x=torch.linspace(-1 * max_x,max_x,100),
            y=torch.linspace(-1 * max_x,max_x,100) * -1,
            mode='lines',
            name='y = -x',
            line = dict(dash = 'dash')
        ))

        component = heads if heads is not None else mlp_layers
        fig.update_layout(
            title=f"Change in Final Logits vs Direct Effect for {component} in {model_name} for each Position and Batch. zero_ablate = {zero_ablate}" if heads is not None 
            else f"Change in Final Logits vs Direct Effect for MLP Layer {component} in {model_name} for each Position and Batch. zero_ablate = {zero_ablate}",
        )
        fig.update_xaxes(title = f"Direct Effect of Head {heads[0]}" if heads is not None else f"Direct Effect of MLP Layer {mlp_layers[0]}")
        fig.update_yaxes(title = "Change in Final Logits")
        fig.update_layout(width=900, height=500)
        fig.show()
    
    if return_slope:
        return slope
def get_threshold_from_percent(logits, threshold_percent_filter):
    logits_flat = logits.flatten()
    sorted_logits = logits_flat.sort()[0]
    index = int((1 - threshold_percent_filter) * sorted_logits.size(0))
    threshold_value = sorted_logits[index]
    return threshold_value
def get_top_self_repair_prompts(heads = None, mlp_layers = None, topk = 10, num_runs = 5, logits = logits, logit_diff_self_repair = True, threshold_percent_filter = 1):
    """
    if logit_diff_self_repair, Top self repair is calcualted by seeing how little the logits change.
    if not, it just calculated by whichever prompts, when ablating the component, changes the most positively in logits.

    threshold_filter controls for which examples to consider; if not zero, it will only consider examples where the absolute direct effect of the component is at least threshold_filter
    """
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        direct_effect = per_head_direct_effect[layer, head]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
        direct_effect = all_layer_direct_effect[layer]
    else:
        raise Exception("No heads or mlp layers given")
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    # get change in direct effect compared to original
    change_in_direct_effect = new_logits_after_sample_wrapper(heads, mlp_layers, num_runs, logits) - get_correct_logit_score(logits)
    
    # get topk
    threshold_filter = get_threshold_from_percent(direct_effect.abs(), threshold_percent_filter)
    mask_direct_effect: Float[Tensor, "batch pos"] = direct_effect.abs() > threshold_filter
    
    if logit_diff_self_repair:
        # Using masked_select to get the relevant values based on the mask.
        change_in_direct_effect = change_in_direct_effect.masked_fill(~mask_direct_effect, 99999)
        topk_indices = topk_of_Nd_tensor(-1 * change_in_direct_effect.abs(), k = topk) # -1 cause we want the minimim change in logits
    else:
        change_in_direct_effect = change_in_direct_effect.masked_fill(~mask_direct_effect, -99999)
        topk_indices = topk_of_Nd_tensor(change_in_direct_effect, k = topk)
    return topk_indices







# %% 

def percent_inductiony_prompts(prompts):
    """
    finds out what percentage of prompts is inductiony;
    prompts should be a tensor of [prompt, 2] where each entry is one of [batch, pos]
    """
    count = 0.0
    for batch, pos in prompts:
        prev = all_owt_tokens[batch, 0:(pos+1)]
        actual = all_owt_tokens[batch, (pos+1):(pos + 2)].item()

        # see if actual is in prev
        if actual in prev:
            count += 1.0
        
            


    return count / len(prompts)


# %%
topk = 40
zldtps = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
mldtps = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        zero_logit_diff_top_prompts = get_top_self_repair_prompts(heads = [(layer, head)], topk = topk, logit_diff_self_repair = True, threshold_percent_filter = 0.05, num_runs=5)
        max_logit_diff_top_prompts = get_top_self_repair_prompts(heads = [(layer, head)], topk = topk, logit_diff_self_repair = False, threshold_percent_filter = 0.05, num_runs=5)


        zldtps[layer, head] = percent_inductiony_prompts(zero_logit_diff_top_prompts)#set([(i[0], i[1]) for i in zero_logit_diff_top_prompts])
        mldtps[layer, head] = percent_inductiony_prompts(max_logit_diff_top_prompts)#set([(i[0], i[1]) for i in max_logit_diff_top_prompts])
        

# %%



# %%
imshow(zldtps, title = f"Percent Inductiony Prompts for topk = {topk} of Zero-Logit-Diff Self Repair in {safe_model_name}", text_auto = True, width = 800, height = 800, range_color=[-1, 1])
imshow(mldtps, title = f"Percent Inductiony Prompts for topk = {topk} of Max-Logit-Diff Self Repair in {safe_model_name}", text_auto = True, width = 800, height = 800, range_color=[-1, 1])

# %%
imshow(mldtps - zldtps, title = "diff of smth", text_auto = True, width = 800, height = 800)


# %%
mlp_zldtps = torch.zeros((model.cfg.n_layers))
mlp_mldtps = torch.zeros((model.cfg.n_layers))
for layer in tqdm(range(model.cfg.n_layers)):
    zero_logit_diff_top_prompts = get_top_self_repair_prompts(mlp_layers = [layer], topk = topk, logit_diff_self_repair = True, threshold_percent_filter = 0.05, num_runs=5)    
    max_logit_diff_top_prompts = get_top_self_repair_prompts(mlp_layers = [layer], topk = topk, logit_diff_self_repair = False, threshold_percent_filter = 0.05, num_runs=5)

    mlp_zldtps[layer] = percent_inductiony_prompts(zero_logit_diff_top_prompts)
    mlp_mldtps[layer] = percent_inductiony_prompts(max_logit_diff_top_prompts)

# %%
imshow(mlp_zldtps.unsqueeze(-1), title = f"Percent Inductiony Prompts for topk = {topk} of Zero-Logit-Diff Self Repair in {safe_model_name}", text_auto = True, width = 200, height = 800)
imshow(mlp_mldtps.unsqueeze(-1), title = f"Percent Inductiony Prompts for topk = {topk} of Max-Logit-Diff Self Repair in {safe_model_name}", text_auto = True, width = 200, height = 800)

# %%
# 
#     


# %% Find a baseline for the inductiony prompts
for j in range(12):
    baseline_prompts = [[i, int(PROMPT_LEN) - j] for i in range(BATCH_SIZE)]
    print(percent_inductiony_prompts(torch.tensor(baseline_prompts)))


# %%

#create_scatter_of_change_from_component(heads = [(9,9)], force_through_origin=False)
layer = 11
for i in range(12):
    head = i
    create_scatter_of_change_from_component(heads = [(layer, head)], zero_ablate=False, force_through_origin=True, num_runs = 5)
#


# %%
layer = 1
head = 8
# for heads
create_scatter_of_change_from_component(heads = [(layer, head)], zero_ablate=False, force_through_origin=True, num_runs = 5)
zero_logit_diff_top_prompts = get_top_self_repair_prompts(heads = [(layer, head)], topk = topk, logit_diff_self_repair = True, threshold_percent_filter = 0.05, num_runs=5)
max_logit_diff_top_prompts = get_top_self_repair_prompts(heads = [(layer, head)], topk = topk, logit_diff_self_repair = False, threshold_percent_filter = 0.05, num_runs=5)

# for mlp layers
# create_scatter_of_change_from_component(mlp_layers= [layer], zero_ablate=False, force_through_origin=True, num_runs = 5)
# zero_logit_diff_top_prompts = get_top_self_repair_prompts(mlp_layers= [layer], topk = topk, logit_diff_self_repair = True, threshold_percent_filter = 0.05, num_runs=5)
# max_logit_diff_top_prompts = get_top_self_repair_prompts(mlp_layers= [layer], topk = topk, logit_diff_self_repair = False, threshold_percent_filter = 0.05, num_runs=5)



# %%
for batch, pos in zero_logit_diff_top_prompts:
    print(f"\n------ new prompt B{batch}P{pos}: ------   head direct effect = {per_head_direct_effect[layer, head, batch, pos]}\n\n")
    
    print_tokens(batch, pos + 1, pos + 2)
    

# %%

for batch, pos in max_logit_diff_top_prompts:
    print(f"\n------ new prompt B{batch}P{pos}: ------   head direct effect = {per_head_direct_effect[layer, head, batch, pos]}\n\n")
    
    print_tokens(batch, pos, pos + 4)



# %%

slopes_of_head_backup = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        slopes_of_head_backup[layer, head] = create_scatter_of_change_from_component(heads = [(layer, head)], return_slope = True, zero_ablate = False, force_through_origin=True, num_runs=5).item()
        


fig = imshow(slopes_of_head_backup, title = f"Slopes of Component Influence on Total Change in Final Logits in {model_name}",
       text_auto = True, width = 800, height = 800, return_fig=True) # show a number above each square)
if in_notebook_mode: 
    fig.show()


# %% ##Get top examples of self-repair
off = 0
tops = topk_of_Nd_tensor(slopes_of_head_backup[off:], k = 4)
print(tops)

if True:
    for layer, head in tops:
        create_scatter_of_change_from_component(heads = [(layer + off, head)], force_through_origin=True, num_runs=5)

# %%
# Do the same thing but for MLP layers
slopes_of_mlp_backup = torch.zeros((model.cfg.n_layers))
for layer in tqdm(range(model.cfg.n_layers)):
    slopes_of_mlp_backup[layer] = create_scatter_of_change_from_component(mlp_layers = [layer], return_slope = True, zero_ablate = True)

# %%
fig = imshow(einops.repeat(slopes_of_mlp_backup, "a -> a 1"), title = f"Slopes of MLP Backup in {model_name}",
       text_auto = True, width = 800, height = 800, return_fig=True)# show a number above each square)

if in_notebook_mode: 
    fig.show()
# fig.write_image(f"slopes_of_mlp_backup_{safe_model_name}.png")
# %%

if in_notebook_mode:
    layer = 4
    head = 6
    temp_head_effect, temp_mlp_effect = dir_effects_from_sample_ablating(attention_heads = [(layer, head)])
    show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)),
            title = f"Logit Diff Diff of Downstream Components upon ablation of {layer}.{head}")

    # layer = 10
    # temp_head_effect, temp_mlp_effect = dir_effects_from_sample_ablating(mlp_layers = [10])
    # show_input(temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2)), temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2)),
    #         title = f"Logit Diff Diff of Downstream Components upon ablation of layer {layer}")
    
    #hist(all_layer_direct_effect[-1].flatten().cpu())



# %% -- Observation; it seems like the direct effects of layer 11 are strangely positive.
flattened_tensors = [per_head_direct_effect[i].flatten().cpu() for i in range(model.cfg.n_layers)]
df = pd.DataFrame({
    'Direct_Effect': np.concatenate(flattened_tensors),
    'Layer': np.concatenate([[f'Layer_{i}'] * len(tensor) for i, tensor in enumerate(flattened_tensors)])
})

# Create the histogram
fig = px.histogram(df, x="Direct_Effect", color="Layer", barmode="overlay", nbins=400)

# Set the range of the histogram


# Add titles
fig.update_layout(
    title="Direct Effects Across Layers",
    xaxis_title="Direct Effect Value",
    yaxis_title="Frequency",
    barmode='overlay'
)
fig.update_xaxes(range=[-3, 7])
fig.update_traces(xbins=dict(start=-3, end=7, size=0.1))

# Show the plot
fig.show()

# %%

def get_threholded_de_cre(heads = None, mlp_layers = None, thresholds = [0.5]):
    """"
    this function calculates direct effect of component on clean and sample ablation,
    and then returns an averaged direct effect and compensatory response effect of the component
    of 'significant' tokens, defined as tokens with a direct effect of at least threshold 


    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    """

    # don't accept if more than one input is none
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
    else:
        raise Exception("No heads or mlp layers given")
    
    ablated_per_head_batch_direct_effect, mlp_per_layer_direct_effect = dir_effects_from_sample_ablating(attention_heads=heads, mlp_layers=mlp_layers)

    # 2) gets the total accumulated backup of the head for each prompt and position
    downstream_change_in_logit_diff: Float[Tensor, "layer head batch pos"] = ablated_per_head_batch_direct_effect - per_head_direct_effect
    downstream_change_in_mlp_logit_diff: Float[Tensor, "layer batch pos"] = mlp_per_layer_direct_effect - all_layer_direct_effect
    

    if (downstream_change_in_logit_diff[0:layer].sum((0,1,2,3)).item() != 0):
        print("assymetry in direct effects due to changed LN scaling alert!")
    if heads is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer):].sum(0)
        total_backup = head_backup + mlp_backup
    if mlp_layers is not None:
        head_backup = downstream_change_in_logit_diff[(layer+1):].sum((0,1))
        mlp_backup = downstream_change_in_mlp_logit_diff[(layer+1):].sum(0)
        total_backup = head_backup + mlp_backup
    

    # 3) filter for indices wherer per_head_direct_effect is greater than threshold
    to_return = {}
    for threshold in thresholds:
        mask_direct_effect: Float[Tensor, "batch pos"] = per_head_direct_effect[layer, head].abs() > threshold
        # Using masked_select to get the relevant values based on the mask.
        selected_cre = total_backup.masked_select(mask_direct_effect)
        selected_de = per_head_direct_effect[layer, head].masked_select(mask_direct_effect)
        
    
        to_return[f"de_{threshold}"] = selected_de.mean().item()
        to_return[f"cre_{threshold}"] = selected_cre.mean().item()
        to_return[f"num_thresholded_{threshold}"] = selected_cre.shape[0]
        
    return to_return


def get_thresholded_change_in_logits(heads = None, mlp_layers = None, thresholds = [0.5]):
    """"
    this function calculates direct effect of component on clean and sample ablation,
    and then returns an averaged direct effect and compensatory response effect of the component
    of 'significant' tokens, defined as tokens with a direct effect of at least threshold 


    heads: list of tuples of (layer, head) to ablate
        - all heads need to be in same layer for now
    """

    # don't accept if more than one input is none
    assert sum([heads is not None, mlp_layers is not None]) == 1

    # make sure all heads are in same layer
    if heads is not None:
        assert len(set([layer for (layer, head) in heads])) == 1
        layer = heads[0][0]
        head = heads[0][1]
        #print(layer)
    elif mlp_layers is not None:
        # layer is max of all the layers
        layer = max(mlp_layers)
    else:
        raise Exception("No heads or mlp layers given")
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]


    change_in_logits = new_logits_after_sample_wrapper(heads, mlp_layers) - get_correct_logit_score(logits)
    # 3) filter for indices wherer per_head_direct_effect is greater than threshold
    to_return = {}
    for threshold in thresholds:
        mask_direct_effect: Float[Tensor, "batch pos"] = (per_head_direct_effect[layer, head].abs() > threshold) if heads != None else (all_layer_direct_effect[layer].abs() > threshold)
        # Using masked_select to get the relevant values based on the mask.
        selected_cil = change_in_logits.masked_select(mask_direct_effect)
        selected_de = per_head_direct_effect[layer, head].masked_select(mask_direct_effect) if heads != None else all_layer_direct_effect[layer].masked_select(mask_direct_effect)
        
    
        to_return[f"de_{threshold}"] = selected_de.mean().item()
        to_return[f"cil_{threshold}"] = selected_cil.mean().item()
        to_return[f"num_thresholded_{threshold}"] = selected_cil.shape[0]
        
    return to_return


def plot_thresholded_de_vs_cre(threshold_de, threshold_cre, threshold_count):
       
    layers_list = [l for l in range(model.cfg.n_layers) for _ in range(model.cfg.n_heads)]
    
    fig = go.Figure()

    for i, threshold in enumerate(thresholds):
        visible = (i == 0)  # Only the first threshold is visible initially
        
        scatter_trace = go.Scatter(
            x=thresholded_de[i].flatten(),
            y=thresholded_cre[i].flatten(),
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.5,
                line=dict(width=1),
                color=layers_list,
                colorscale='Viridis',
                colorbar=dict(title='Layer', y=10),
            ),
            text=[f"Layer {l}, Head {h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
            visible=visible,
            name=f"Threshold: {threshold}"
        )

        line_trace = go.Scatter(
            x=torch.linspace(-2, 2, 100),
            y=torch.linspace(-2, 2, 100),
            mode='lines',
            line=dict(dash='dash'),
            visible=visible
        )
        
        fig.add_trace(scatter_trace)
        fig.add_trace(line_trace)
    
    frames = []
    for i, threshold in enumerate(thresholds):
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=thresholded_de[i].flatten().numpy(),
                    y=thresholded_cre[i].flatten().numpy(),
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.5,
                        line=dict(width=1),
                        color=layers_list,
                        colorscale='Viridis',
                        colorbar=dict(title='Layer', y=10),
                    ),
                    text=[f"Layer {l}, Head {h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)],
                )
            ],
            name=str(threshold)   # Use the threshold as the name of the frame
        )
        frames.append(frame)

    
    fig.frames = frames

    
    
    steps = []
    for i, threshold in enumerate(thresholds):
        step = {
            'args': [
                [str(threshold)],  # Frame name to show
                {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 300}}
            ],
            'label': str(threshold),
            'method': 'animate'
        }
        steps.append(step)


    sliders = [dict(
        yanchor='top',
        xanchor='left',
        currentvalue={'font': {'size': 16}, 'prefix': 'Threshold: ', 'visible': True, 'xanchor': 'right'},
        transition={'duration': 300, 'easing': 'cubic-in-out'},
        pad={'b': 10, 't': 50},
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=f"Thresholded Compensatory Response Effect vs Average Direct Effect in {model_name}",
        xaxis_title="Average Direct Effect",
        yaxis_title="Average Compensatory Response Effect",
        hovermode="closest",
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        width=1000,
        height=400,
        showlegend=False,
    )
    
    if in_notebook_mode:
        fig.show()
    
    fig.write_html(f"threshold_figures/threshold_{safe_model_name}_cre_vs_avg_direct_effect_slider.html")


def gpt_new_plot_thresholded_de_vs_cre(thresholded_de, thresholded_cre, thresholds, use_logits = False):
    """
    create a plot of the self-repair against the direct effect, for various heads and components. if use_logits = True, then we will use
    th change in logits instead of the compensatory response effect, and threshold_cre should be threshold_cil
    """
    # Generate a list of (layer, head) tuples for sorting
    layer_head_list = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    x_labels = [f"L{layer}H{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    
    fig = go.Figure()

    for i, threshold in enumerate(thresholds):
        visible = (i == 0)  # Only the first threshold is visible initially

        # Sort based on layer and head
        sorted_indices = sorted(range(len(layer_head_list)), key=lambda k: layer_head_list[k])
        sorted_de = thresholded_de[i].flatten()[sorted_indices]
        sorted_cre = thresholded_cre[i].flatten()[sorted_indices]

        scatter_trace = go.Scatter(
            x=x_labels,  # x-axis is just indices after sorting
            y=sorted_cre.numpy(),  # Sorted CRE values
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.5,
                line=dict(width=1),
                color=sorted_de.numpy(),  # Color by direct effect
                colorscale='RdBu',  # Red to Blue scale
                colorbar=dict(title='Direct Effect', y=10),
                cmin=-max(abs(sorted_de.numpy())),
                cmax=max(abs(sorted_de.numpy()))
            ),
            text=[f"Layer {layer_head_list[i][0]}, Head {layer_head_list[i][1]}: DE = {sorted_de[i]}" for i in sorted_indices],
            visible=visible,
            name=f"Threshold: {threshold}"
        )
        
        fig.add_trace(scatter_trace)
    
    frames = []
    for i, threshold in enumerate(thresholds):
        sorted_indices = sorted(range(len(layer_head_list)), key=lambda k: layer_head_list[k])
        sorted_de = thresholded_de[i].flatten()[sorted_indices]
        sorted_cre = thresholded_cre[i].flatten()[sorted_indices]

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x_labels,
                    y=sorted_cre.numpy(),
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.5,
                        line=dict(width=1),
                        color=sorted_de.numpy(),
                        colorscale='RdBu',
                        colorbar=dict(title='Direct Effect', y=10),
                        cmin=-max(abs(sorted_de.numpy())),
                        cmax=max(abs(sorted_de.numpy()))
                    ),
                    text=[f"Layer {layer_head_list[i][0]}, Head {layer_head_list[i][1]}: DE = {sorted_de[i]}" for i in sorted_indices],
                )
            ],
            name=str(threshold)
        )
        frames.append(frame)

    fig.frames = frames
    
    steps = []
    for i, threshold in enumerate(thresholds):
        sorted_indices = sorted(range(len(layer_head_list)), key=lambda k: layer_head_list[k])
        sorted_cre = thresholded_cre[i].flatten()[sorted_indices]
        
        # Calculate the y-axis range for this threshold
        y_min = sorted_cre.min().item() - 0.1  # Adding some padding
        y_max = sorted_cre.max().item() + 0.1
        print(y_min, y_max)

        step = {
            'args': [
                [str(threshold)],  # Frame name to show
                {
                    'frame': {'duration': 300, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 300},
                    'relayout': {'yaxis.range': [y_min, y_max]}  # Update y-axis range
                }
            ],
            'label': str(threshold),
            'method': 'animate'
        }
        steps.append(step)


    sliders = [dict(
        yanchor='top',
        xanchor='left',
        currentvalue={'font': {'size': 16}, 'prefix': 'Threshold: ', 'visible': True, 'xanchor': 'right'},
        transition={'duration': 300, 'easing': 'cubic-in-out'},
        pad={'b': 10, 't': 50},
        len=0.9,
        x=0.1,
        y=0,
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        title=f"Thresholded Compensatory Response Effect vs Average Direct Effect in {model_name}" if not use_logits else f"Thresholded Change in Output Logits vs Average Direct Effect in {model_name}",
        xaxis_title="Average Direct Effect",
        yaxis_title="Average Compensatory Response Effect",
        hovermode="closest",
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {'frame': {'duration': 500, 'redraw': True}, 'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'showactive': False,
            'type': 'buttons',
            'x': 0.1,
            'xanchor': 'right',
            'y': 0,
            'yanchor': 'top'
        }],
        width=1000,
        height=400,
        showlegend=False,
    )
    if in_notebook_mode:
        fig.show()
    
    fig.write_html(f"threshold_figures/threshold_{safe_model_name}_cre_vs_avg_direct_effect_slider.html")

    


# %%

# thresholds = [0.1 * i for i in range(0,12)]

# thresholded_de = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))
# thresholded_cre = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))
# thresholded_count = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))

# for layer in tqdm(range(model.cfg.n_layers)):
#     for head in range(model.cfg.n_heads):
#         dict_results = get_threholded_de_cre(heads = [(layer, head)], thresholds = thresholds)
#         for i, threshold in enumerate(thresholds):
#             thresholded_de[i, layer, head] = dict_results[f"de_{threshold}"]
#             thresholded_cre[i, layer, head] = dict_results[f"cre_{threshold}"]
#             thresholded_count[i, layer, head] = dict_results[f"num_thresholded_{threshold}"]


# %%
thresholds = [0.1 * i for i in range(0,12)]
thresholded_de = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))
thresholded_cil = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))
thresholded_count = torch.zeros((len(thresholds), model.cfg.n_layers, model.cfg.n_heads))



for layer in tqdm(range(model.cfg.n_layers)):
    for head in range(model.cfg.n_heads):
        dict_results = get_thresholded_change_in_logits(heads = [(layer, head)], thresholds = thresholds)
        for i, threshold in enumerate(thresholds):
            thresholded_de[i, layer, head] = dict_results[f"de_{threshold}"]
            thresholded_cil[i, layer, head] = dict_results[f"cil_{threshold}"]
            thresholded_count[i, layer, head] = dict_results[f"num_thresholded_{threshold}"]


#plot_thresholded_de_vs_cre(thresholded_de, thresholded_cre, thresholded_count)

# %%
gpt_new_plot_thresholded_de_vs_cre(thresholded_de, thresholded_cil, thresholds, True)

# %%
#plot_thresholded_de_vs_cre([-99999999999999999])
#plot_thresholded_de_vs_cre([0.1 * i for i in range(0,12)])
# %%


# %%





# %% graveyard:
# ----------------------------------------------------------------
# Get the extent to which heads can have downstream heads act positively, rather than negatively
# pairs_of_head_backup = {}
# pairs_of_mlp_backup = {}
# threshold = 0.05
# for layer in tqdm(range(model.cfg.n_layers)):
#     for head in range(model.cfg.n_heads):
#         temp_head_effect, temp_mlp_effect = dir_effects_from_sample_ablating(attention_heads = [(layer, head)])
#         head_backup: Float[Tensor, "layer head"] = temp_head_effect.mean((-1,-2)) - per_head_direct_effect.mean((-1,-2))
#         mlp_backup: Float[Tensor, "layer"] = temp_mlp_effect.mean((-1,-2)) - all_layer_direct_effect.mean((-1,-2))

#         if (head_backup > threshold).sum() > 0:
#             pairs_of_head_backup[(layer, head)] = True
#         else:
#             pairs_of_head_backup[(layer, head)] = False

#         if (mlp_backup > threshold).sum() > 0:
#             pairs_of_mlp_backup[layer] = True
#         else:
#             pairs_of_mlp_backup[layer] = False

# for layer, head in pairs_of_head_backup.keys():
#     if pairs_of_head_backup[(layer, head)]:
#         print((layer, head))

# for layer in range(12):
#     if pairs_of_mlp_backup[layer]:
#         print(layer)


# ----------------------------------------------------------------

# # RED TEAM EARLIER ESULT

# new_logits = act_patch(model, owt_tokens,  [Node("z", 1, 8)], new_input = corrupted_owt_tokens, patching_metric = return_item,)
# new_probs = torch.softmax(new_logits, -1)
# old_probs = torch.softmax(logits, -1)

# values_new, indices_new = torch.topk(new_probs[53, 12, :], 5)
# print("Top 5 values of new_logits[53, 12, :]:", values_new)
# print("Top 5 indices of new_logits[53, 12, :]:", indices_new)


# # For logits
# values, indices = torch.topk(old_probs[53, 12, :], 5)
# print("Top 5 values of logits[53, 12, :]:", values)
# print("Top 5 indices of logits[53, 12, :]:", indices)


# ablated_1_8_cache = act_patch(model, owt_tokens,  [Node("z", 1, 8)],
#                             return_item, corrupted_owt_tokens, apply_metric_to_cache= True)


# for layer in range(model.cfg.n_layers):
#     print(ablated_1_8_cache[utils.get_act_name("resid_pre", layer)].shape)

# head_de, mlp_layer_de = collect_direct_effect(ablated_1_8_cache, owt_tokens, display = True, collect_individual_neurons = False)
# show_input(head_de[..., 53,12], mlp_layer_de[..., 53,12], title = "Direct Effect of Ablating Head 1.8")


# show_input(per_head_direct_effect[..., 53,12], all_layer_direct_effect[..., 53,12], title = "Direct Effect of Ablating Head 1.8")

# token_residual_directions: Float[Tensor, "batch seq_len d_model"] = model.tokens_to_residual_directions(owt_tokens)
# all_layer_output: Float[Tensor, "n_layer batch pos d_model"] = torch.zeros((model.cfg.n_layers, owt_tokens.shape[0], owt_tokens.shape[1], model.cfg.d_model)).cuda()
# for layer in range(model.cfg.n_layers):
#     all_layer_output[layer, ...] = ablated_1_8_cache[f'blocks.{layer}.hook_mlp_out']


# scaled_residual_stack = ablated_1_8_cache.apply_ln_to_stack(all_layer_output, layer = -1, has_batch_dim = True)
# scaled_residual_stack = scaled_residual_stack[..., :, :-1, :]
# effect_directions = token_residual_directions[:, 1:, :]
# dot_prods = einops.einsum(scaled_residual_stack, effect_directions, "... batch pos d_model, batch pos d_model -> ... batch pos")

# dot_prods[-1, 53, 12]
# %%
