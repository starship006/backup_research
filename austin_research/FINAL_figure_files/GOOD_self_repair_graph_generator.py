"""
This code is responsible for making the self-repair graphs.
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/new_graph_pickle/"
NO_PADDING = True

if in_notebook_mode:
    model_name = "gpt2-medium"#"pythia-160m"####
    BATCH_SIZE = 2
    ABLATION_TYPE = "sample" 
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--ablation_type', type=str, default='mean', choices=['mean', 'zero', 'sample'])
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size
    ABLATION_TYPE = args.ablation_type

# Ensure that ABLATION_TYPE is one of the expected values
assert ABLATION_TYPE in ["mean", "zero", "sample"], "Ablation type must be 'mean', 'zero', or 'sample'."

# %% Import the Model
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
model.set_use_attn_result(False)


# %%
dataset = utils.get_dataset("pile")
dataset_name = "The Pile"

PROMPT_LEN = 200

if NO_PADDING:
    new_dataset = utils.tokenize_and_concatenate(dataset, model.tokenizer, max_length = PROMPT_LEN) #type: ignore
    all_dataset_tokens = new_dataset['tokens'].to(device) #type: ignore
else:
    all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
# %% Helper Functions
def new_logits_upon_ablation_calc(clean_tokens, corrupted_tokens, logits: Tensor, heads = None, mlp_layers = None, num_runs = 1, clean_cache: Union[None, ActivationCache] = None):
    """
    gets the logits of correct token when running the model on some tokens and ablating some heads or mlp layers
    averaged over num_runs different ablations (by default, only just 1 run. this may need to be greater if working with smaller batch sizes)
    
    clean_cache for mean ablation
    """
    
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]
    if ABLATION_TYPE == "zero" or ABLATION_TYPE == "mean":
        num_runs = 1 # since zero or mean ablation would be deterministic
    
    logits_accumulator = torch.zeros_like(logits, device=logits.device)
    for _ in range(num_runs):
        # Shuffle owt_tokens by batch
        shuffled_corrupted_tokens = shuffle_owt_tokens_by_batch(corrupted_tokens)
        # Calculate new_logits using act_patch
        if ABLATION_TYPE == "zero":
            new_logits = act_patch(model, clean_tokens, nodes, return_item, new_cache = "zero", apply_metric_to_cache=False)# type: ignore
        elif ABLATION_TYPE == "mean":
            assert clean_cache is not None, "clean_cache must be provided for mean ablation"
            assert len(nodes) == 1 and mlp_layers == None and heads != None, "mean ablation currently only works for one head"
            
            # get average output of layer
            avg_output_of_layer = clean_cache[utils.get_act_name("z", heads[0][0])][:, :, heads[0][1], :].mean((0,1))
            #W_U = model.W_O[heads[0][0], heads[0][1]]
            #avg_output_of_layer = einops.einsum(avg_output_of_layer, W_U, "d_head, d_head d_model -> d_model")
            avg_output_of_layer = einops.repeat(avg_output_of_layer, "d_model -> batch seq d_model", batch = clean_tokens.shape[0], seq = clean_tokens.shape[1])
            
            # run with hook which mean ablates
            model.reset_hooks()
            hook = partial(replace_output_hook, new_output = avg_output_of_layer, head = heads[0][1])
            model.add_hook(utils.get_act_name("z", heads[0][0]), hook)
            new_logits = model(clean_tokens)
            model.reset_hooks()
        else:
            new_logits = act_patch(model, clean_tokens, nodes, return_item, shuffled_corrupted_tokens, apply_metric_to_cache=False)
        logits_accumulator += new_logits

    avg_logits = logits_accumulator / num_runs
    
    # get change in direct effect compared to original
    avg_correct_logit_score = get_correct_logit_score(avg_logits, clean_tokens)
    return avg_correct_logit_score

def get_thresholded_change_in_logits(clean_tokens, corrupted_tokens, logits: Tensor, per_component_direct_effect: Tensor, heads = None, mlp_layers = None, thresholds = [0.5], cache: Union[None, ActivationCache] = None):
    """"
    this function calculates direct effect of component on clean runs and ablations,
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
        head = None
    else:
        raise ValueError("No heads or mlp layers given")
    
    nodes = []
    if heads is not None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers is not None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]

    
    change_in_logits = new_logits_upon_ablation_calc(clean_tokens, corrupted_tokens, logits, heads, mlp_layers, clean_cache=cache) - get_correct_logit_score(logits, clean_tokens)
        
    # 3) filter for indices wherer per_component_direct_effect is greater than threshold
    to_return = {}
    for threshold in thresholds:
        if heads != None:
            mask_direct_effect: Float[Tensor, "batch pos"] = (per_component_direct_effect[layer, head].abs() > threshold)
        else:
            mask_direct_effect: Float[Tensor, "batch pos"] = (per_component_direct_effect[layer].abs() > threshold)
        
        # Using masked_select to get the relevant values based on the mask.
        selected_cil = change_in_logits.masked_select(mask_direct_effect)
        
        if heads != None:
            selected_de = per_component_direct_effect[layer, head].masked_select(mask_direct_effect)
        else:
            selected_de = per_component_direct_effect[layer].masked_select(mask_direct_effect)
        
        
        to_return[f"de_{threshold}"] = selected_de.mean().item()
        to_return[f"cil_{threshold}"] = selected_cil.mean().item()
        to_return[f"num_thresholded_{threshold}"] = selected_cil.shape[0]
    return to_return

# %% We need to iterate through the dataset
TOTAL_PROMPTS_TO_ITERATE_THROUGH = 240 #around 500 / 2

# BATCH_SIZE defined earlier

num_batches = TOTAL_PROMPTS_TO_ITERATE_THROUGH // BATCH_SIZE
assert model.tokenizer is not None
PAD_TOKEN = model.to_tokens(model.tokenizer.pad_token)[-1, -1].item() 
print("Percent of tokens that are padding tokens: ", (all_dataset_tokens[:(TOTAL_PROMPTS_TO_ITERATE_THROUGH + BATCH_SIZE), :PROMPT_LEN] == PAD_TOKEN).flatten().float().mean(-1))
# %% We filter out for specific instances where the direct effects are above a certain amount. By default, the figures in the paper don't worry about thresholds, but they are nice to have
THRESHOLDS = [0.0] #[0.1 * i for i in range(0,12)]
thresholded_de = torch.zeros((num_batches, len(THRESHOLDS), model.cfg.n_layers, model.cfg.n_heads))
thresholded_cil = torch.zeros((num_batches, len(THRESHOLDS), model.cfg.n_layers, model.cfg.n_heads))
thresholded_count = torch.zeros((num_batches, len(THRESHOLDS), model.cfg.n_layers, model.cfg.n_heads))

# %% Run. This takes a while.
for batch in tqdm(range(num_batches)):
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
    assert isinstance(logits, Tensor)
    
    #corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
    per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens, model = model, display = False, collect_individual_neurons = False)
    
    #if in_notebook_mode:
    #    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")
    
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            dict_results = get_thresholded_change_in_logits(clean_tokens, corrupted_tokens, logits, per_head_direct_effect, heads = [(layer, head)], thresholds = THRESHOLDS, cache = cache)
            for i, threshold in enumerate(THRESHOLDS):
                thresholded_de[batch, i, layer, head] = dict_results[f"de_{threshold}"]
                thresholded_cil[batch, i, layer, head] = dict_results[f"cil_{threshold}"]
                thresholded_count[batch, i, layer, head] = dict_results[f"num_thresholded_{threshold}"]

# %% Average across batches
thresholded_de = thresholded_de.mean(0)
thresholded_cil = thresholded_cil.mean(0)
thresholded_count = thresholded_count.mean(0)

# %% Plotting functionality
def gpt_new_plot_thresholded_de_vs_cre(thresholded_de, thresholded_cre, thresholds, use_logits = False, layout_horizontal = True):
    """
    create a plot of the self-repair against the direct effect, for various heads and components. if use_logits = True, then we will use
    th change in logits instead of the compensatory response effect, and threshold_cre should be threshold_cil

    layout_horizontal: whether or not to do the 'each head side by side' threshold plot
    """
    # Generate a list of (layer, head) tuples for sorting
    layer_head_list = [(l, h) for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)]
    
    
    fig = go.Figure()
    x_labels = [f"L{layer}H{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
    sorted_indices = sorted(range(len(layer_head_list)), key=lambda k: layer_head_list[k])
    
    num_layers = model.cfg.n_layers
    num_heads = model.cfg.n_heads
    layer_colors = np.linspace(0, num_layers, num_layers, endpoint = False)
    head_annotations = [f"Layer {layer}, Head {head}" for layer, head in itertools.product(range(num_layers), range(num_heads))]
    head_marker_colors = [layer_colors[layer] for layer in range(num_layers) for _ in range(num_heads)]


    #print(head_marker_colors)
    for i, threshold in enumerate(thresholds):
        visible = (i == 0)  # Only the first threshold is visible initially

        # Sort based on layer and head
        sorted_de = thresholded_de[i].flatten()[sorted_indices]
        sorted_cre = thresholded_cre[i].flatten()[sorted_indices]
        if layout_horizontal:
            x_values = x_labels
            y_values = sorted_cre.numpy()
            color_values = sorted_de.numpy()
            colorscale='RdBu'
            cmin = -max(abs(sorted_de.numpy()))
            cmax = max(abs(sorted_de.numpy()))
        else:
            x_values = sorted_de.numpy()
            y_values = sorted_cre.numpy()
            color_values = head_marker_colors
            colorscale='Viridis'
            cmin = 0
            cmax = num_layers - 1

        scatter_trace = go.Scatter(
            x=x_values,  # x-axis is just indices after sorting
            y=y_values,  # Sorted CRE values
            mode='markers',
            marker=dict(
                size=10,
                opacity=0.5,
                line=dict(width=1),
                color=color_values,  # Color by direct effect
                colorscale=colorscale,  # Red to Blue scale
                colorbar=dict(title='Direct Effect', y=10),
                cmin=cmin,
                cmax=cmax,
            ),
            text=[f"Layer {layer_head_list[i][0]}, Head {layer_head_list[i][1]}: DE = {sorted_de[i]}" for i in sorted_indices],
            visible=visible,
            name=f"Threshold: {threshold}"
        )
        
        fig.add_trace(scatter_trace)
    
    frames = []
    steps = []
    for i, threshold in enumerate(thresholds):
        # make the frame
        sorted_de = thresholded_de[i].flatten()[sorted_indices]
        sorted_cre = thresholded_cre[i].flatten()[sorted_indices]

        if layout_horizontal:
            x_values = x_labels
            y_values = sorted_cre.numpy()
            color_values = sorted_de.numpy()
            colorscale='RdBu'
            cmin = -max(abs(sorted_de.numpy()))
            cmax = max(abs(sorted_de.numpy()))
        else:
            x_values = sorted_de.numpy()
            y_values = sorted_cre.numpy()
            color_values = head_marker_colors
            colorscale='Viridis'
            cmin = 0
            cmax = num_layers - 1

    
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='markers',
                    marker=dict(
                        size=10,
                        opacity=0.5,
                        line=dict(width=1),
                        color=color_values,
                        colorscale=colorscale,
                        colorbar=dict(title='Direct Effect', y=10),
                        cmin=cmin,
                        cmax=cmax
                    ),
                    text=[f"Layer {layer_head_list[i][0]}, Head {layer_head_list[i][1]}: DE = {sorted_de[i]}" for i in sorted_indices],
                )
            ],
            name=str(threshold)
        )
        frames.append(frame)

        # make the step
        # Calculate the y-axis range for this threshold
        y_min = y_values.min().item() - 0.1  # Adding some padding
        y_max = y_values.max().item() + 0.1

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

    fig.frames = frames

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
        yaxis_title="Average Change in Logits after Ablation (CRE)",
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
    
    folder = ""
    if layout_horizontal:
        folder = FOLDER_TO_WRITE_GRAPHS_TO + "horizontal_plot_graphs/"
    else:
        folder = FOLDER_TO_WRITE_GRAPHS_TO + "scatter_plot_self_repair_graphs/"
    
    fig.write_html(folder + f"threshold_{safe_model_name}_cre_vs_avg_direct_effect_slider.html")

# %%
#gpt_new_plot_thresholded_de_vs_cre(thresholded_de, thresholded_cil, THRESHOLDS, True, layout_horizontal=False)
#gpt_new_plot_thresholded_de_vs_cre(thresholded_de, thresholded_cil, THRESHOLDS, True, layout_horizontal=True)
# %%
# cappitalize first char
ablation_str = ABLATION_TYPE.capitalize()

fig = create_layered_scatter(thresholded_de[0], thresholded_cil[0], model, "Direct Effect of Component", "Change in Logits Upon Ablation", f"Effect of {ablation_str}-Ablating Attention Heads in {model_name}")
# %%

#type(create_layered_scatter(thresholded_de[0], thresholded_cil[0], model, "Direct Effect of Component", "Change in Logits Upon Ablation", f"Effect of {ablation_type}-Ablating Attention Heads in {model_name}"))
# %%
fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"simple_plot_graphs/{ablation_str}_{safe_model_name}_de_vs_cre.html")

# %% Store the tensors as pickles
type_modifier = "ZERO_" if ABLATION_TYPE == "zero" else ("MEAN_" if ABLATION_TYPE == "mean" else "")
                
# Assuming thresholded_de, thresholded_cil, thresholded_count, model_name are all defined above
thresholds_str = "_".join(map(str, THRESHOLDS))  # Converts thresholds list to a string
# Serialize and save thresholded_de
with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_de_{thresholds_str}.pkl", "wb") as f:
    pickle.dump(thresholded_de, f)
# Serialize and save thresholded_ci
with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_cil_{thresholds_str}.pkl", "wb") as f:
    pickle.dump(thresholded_cil, f)
# Serialize and save thresholded_count
with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_count_{thresholds_str}.pkl", "wb") as f:
    pickle.dump(thresholded_count, f)
# %%
