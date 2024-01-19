# this generates a graph to show that all models self-repair to some extent
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %%
FOLDER_TO_STORE_PICKLES = "pickle_storage/new_graph_pickle/"
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/comparison_graphs/"
ABLATION_TYPE = "sample"

models_to_consider = ["pythia-160m", "gpt2-small", "pythia-410m", "gpt2-medium", "gpt2-large", "pythia-1b", "llama-7b"]
safe_model_names = []
for model_name in models_to_consider:
    safe_model_names.append(model_name.replace("/", "_"))
# %%
thresholded_des = []
thresholded_cils = []
thresholded_counts = []

for safe_model_name in safe_model_names:
    type_modifier = "ZERO_" if ABLATION_TYPE == "zero" else ("MEAN_" if ABLATION_TYPE == "mean" else "")
    THRESHOLDS = [0.0]
    thresholds_str = "_".join(map(str, THRESHOLDS))  # Converts thresholds list to a string
    ablation_str = ABLATION_TYPE.capitalize()
    # LOAD FILES
    with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_de_{thresholds_str}.pkl", "rb") as f:
        thresholded_de = pickle.load(f)[0]
        thresholded_des.append(thresholded_de)
    with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_cil_{thresholds_str}.pkl", "rb") as f:
        thresholded_cil = pickle.load(f)[0]
        thresholded_cils.append(thresholded_cil)
    with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_count_{thresholds_str}.pkl", "rb") as f:
        thresholded_count = pickle.load(f)[0]
        thresholded_counts.append(thresholded_count)
    
# %%
layers_per_model = []
heads_per_layer_per_model = []

for data in thresholded_des:
    layers, heads_per_layer = data.shape
    layers_per_model.append(layers)
    heads_per_layer_per_model.append(heads_per_layer)
# %%
self_repair_percentages = []
self_repair_percentage_per_layer_per_model = []

for i in range(len(thresholded_des)):
    print(f"Model: {models_to_consider[i]}")
    des = thresholded_des[i]
    cils = thresholded_cils[i]
    
    self_repair_amount = cils + des
    
    print(self_repair_amount)
    self_repair_percentage = self_repair_amount / des
    self_repair_percentages.append(self_repair_percentage)
    print(self_repair_percentage)
    # floor all values between 0 and 1
    self_repair_percentage = np.clip(self_repair_percentage, 0, 1)
    self_repair_percentage_per_layer_per_model.append(self_repair_percentage.mean(axis=1))
    #self_repair_percentage_per_layer_per_model.append(data.mean(axis=1))
# %% Using plotly, graph the self-repair percentage per layer per model

fig = go.Figure()
for i in range(len(self_repair_percentage_per_layer_per_model)):
    self_repair_percentage = self_repair_percentage_per_layer_per_model[i]
    fig.add_trace(go.Scatter(x=np.arange(len(self_repair_percentage)), y=self_repair_percentage, mode="lines+markers", name=models_to_consider[i]))
    

fig.update_layout(
    title="Average Self-Repair as a fraction of the Direct Effect Per Layer Across Models",
    xaxis_title="Layer",
    yaxis_title="Self-Repair Percentage",
)

fig.show()

# %%
