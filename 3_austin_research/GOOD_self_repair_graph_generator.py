"""
This code is responsible for making the 'thresholded self-repair' graphs.
"""
# %%
from imports import *
# %%
#%load_ext autoreload
#%autoreload 2

from GOOD_helpers import *
# %% Constants
in_notebook_mode = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if in_notebook_mode:
    model_name = "gpt2-small"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    args = parser.parse_args()
    model_name = args.model_name

BATCH_SIZE = 150
PROMPT_LEN = 30

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
# %%
per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=owt_tokens, model = model, display = in_notebook_mode, collect_individual_neurons = False)
if in_notebook_mode:
    show_input(per_head_direct_effect.mean((-1,-2)), all_layer_direct_effect.mean((-1,-2)), title = "Direct Effect of Heads and MLP Layers")
# %%
def new_ld_upon_sample_ablation_calc(heads = None, mlp_layers = None, num_runs = 5):
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
    avg_correct_logit_score = get_correct_logit_score(avg_logits, owt_tokens)
    return avg_correct_logit_score

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


    change_in_logits = new_ld_upon_sample_ablation_calc(heads, mlp_layers) - get_correct_logit_score(logits, owt_tokens)
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
# %%

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


    print(head_marker_colors)
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
    
    folder = ""
    if layout_horizontal:
        folder = "GOOD_clean_figures/horizontal_plot_graphs/"
    else:
        folder = "GOOD_clean_figures/scatter_plot_self_repair_graphs/"
    
    fig.write_html(folder + f"threshold_{safe_model_name}_cre_vs_avg_direct_effect_slider.html")

    
# %%
gpt_new_plot_thresholded_de_vs_cre(thresholded_de, thresholded_cil, thresholds, True, layout_horizontal=False)
# %%
#create_layered_scatter(thresholded_de[0], thresholded_cil[0], model, "dr", "cre", "de vs cre",)
# %%
