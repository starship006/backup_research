"""
This code is responsible for taking the data needed to make the self-repair graphs, and making them.
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/new_graph_pickle/"
PADDING = False

if in_notebook_mode:
    model_name = "llama-7b"#"pythia-160m"####
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

# %%  Tensors stored as pickles
type_modifier = "ZERO_" if ABLATION_TYPE == "zero" else ("MEAN_" if ABLATION_TYPE == "mean" else "")
THRESHOLDS = [0.0]
thresholds_str = "_".join(map(str, THRESHOLDS))  # Converts thresholds list to a string
ablation_str = ABLATION_TYPE.capitalize()
# LOAD FILES
with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_de_{thresholds_str}.pkl", "rb") as f:
    thresholded_de = pickle.load(f)[0]
with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_cil_{thresholds_str}.pkl", "rb") as f:
    thresholded_cil = pickle.load(f)[0]
with open(FOLDER_TO_STORE_PICKLES + type_modifier + f"{safe_model_name}_thresholded_count_{thresholds_str}.pkl", "rb") as f:
    thresholded_count = pickle.load(f)[0]
# %%
layers, heads = thresholded_de.shape
assert thresholded_de.shape == thresholded_cil.shape == thresholded_count.shape
# %%
fig = create_layered_scatter(thresholded_de, thresholded_cil, model, "Direct Effect of Component", "Change in Logits Upon Ablation", f"Effect of {ablation_str}-Ablating Attention Heads in {model_name}")

fig.add_trace(go.Scatter(x=[min(fig.data[0]['x']) - 0.05, max(fig.data[0]['x']) + 0.05],
                       y=[-min(fig.data[0]['x']) + 0.05, -max(fig.data[0]['x']) - 0.05],
                       mode='lines',
                       name='y=-x Line',
                       line=dict(color='blue', dash='dot'),
                       ))
# %%
fig.update_xaxes(
   showline=True,
   showticklabels=True,
   zeroline=True,
   zerolinecolor='black',
   showgrid=False
)

fig.update_yaxes(
   showline=True,
   showticklabels=True,
   zeroline=True,
   zerolinecolor='black',
   showgrid=False
)
fig.update_layout(
   autosize=False,
   #width=500,
   #height=500,
   margin=dict(
       l=50,
       r=50,
       b=100,
       t=100,
       pad=4
   ),
   showlegend=True
   #paper_bgcolor="rgba(0,0,0,0)",
   #plot_bgcolor="rgba(0,0,0,0)"
)

# Add annotation for 'self-repair' zone
fig.add_annotation(
   x=(min(fig.data[0]['x']) + max(fig.data[0]['x'])) / 2 + 0.2, # middle of x range
   y=(-min(fig.data[0]['x']) + 0.05) / 2 - 0.2, # middle of y range
   text="Self-repair Zone",
   showarrow=False,
   font=dict(size=10, color="black"),
   align="center",
   ax=0,
   ay=-30,
)


fig.show()
# %%

fig.write_html(FOLDER_TO_WRITE_GRAPHS_TO + f"simple_plot_graphs/{ablation_str}_{safe_model_name}_de_vs_cre.html")
fig.write_image(FOLDER_TO_WRITE_GRAPHS_TO + f"simple_plot_graphs/{ablation_str}_{safe_model_name}_de_vs_cre.pdf")

# %%
