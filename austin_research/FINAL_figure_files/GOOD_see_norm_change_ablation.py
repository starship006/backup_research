"""
This code analyzes how the ablation of attention head leads to changes in the residual stream norm.
"""
# %%
from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, show_input, create_layered_scatter, replace_output_hook
# %% Constants
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/"


if in_notebook_mode:
    model_name = "pythia-410m"
    BATCH_SIZE = 15
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt2-small')
    parser.add_argument('--batch_size', type=int, default=30)
    args = parser.parse_args()
    
    model_name = args.model_name
    BATCH_SIZE = args.batch_size

# Ensure that ABLATION_TYPE is one of the expected values

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
model.set_use_attn_result(False)

# %%
dataset = utils.get_dataset("pile")
dataset_name = "The Pile"
all_dataset_tokens = model.to_tokens(dataset["text"]).to(device)
# %% Helper Functions

def get_norm_from_cache(cache: ActivationCache) -> Float[Tensor, "batch pos"]:
    final_resid = cache[utils.get_act_name("resid_post", model.cfg.n_layers - 1)]
    return final_resid.norm(dim = -1)


def new_norm_upon_ablation_calc(clean_tokens, corrupted_tokens, ablation_type: str, heads = None, mlp_layers = None, clean_cache: ActivationCache = None):
    nodes = []
    if heads != None :
        nodes += [Node("z", layer, head) for (layer,head) in heads]
    if mlp_layers != None:
        nodes += [Node("mlp_out", layer) for layer in mlp_layers]
    if ablation_type == "zero" or ablation_type == "mean":
        num_runs = 1 # since zero or mean ablation would be deterministic
    
    # Shuffle owt_tokens by batch
    shuffled_corrupted_tokens = shuffle_owt_tokens_by_batch(corrupted_tokens)
    # Calculate new_logits using act_patch
    new_cache = None
    if ablation_type == "zero":
        new_cache = act_patch(model, clean_tokens, nodes, return_item, new_cache = "zero", apply_metric_to_cache=True)
    elif ablation_type == "mean":
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
        _, new_cache = model.run_with_cache(clean_tokens)
        model.reset_hooks()
    else:
        new_cache = act_patch(model, clean_tokens, nodes, return_item, shuffled_corrupted_tokens, apply_metric_to_cache=True)
      
    assert isinstance(new_cache, ActivationCache)
    return get_norm_from_cache(new_cache)




# %% We need to iterate through the dataset
TOTAL_PROMPTS_TO_ITERATE_THROUGH = 200

# BATCH_SIZE defined earlier
PROMPT_LEN = 400

num_batches = TOTAL_PROMPTS_TO_ITERATE_THROUGH // BATCH_SIZE
# %% Run. This takes a while.
ABLATION_TYPES = ["mean", "zero", "sample"]

for layer in range(model.cfg.n_layers - 1, model.cfg.n_layers):
    for head in [0]:#range(model.cfg.n_heads):
        norm_ratios = torch.zeros((num_batches, len(ABLATION_TYPES), BATCH_SIZE, PROMPT_LEN))

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
            #corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
            
            clean_norm = get_norm_from_cache(cache)
        
            new_norms = []
            
            for ablation_type in ABLATION_TYPES:
                assert ablation_type in ["mean", "zero", "sample"], "Ablation type must be 'mean', 'zero', or 'sample'."
                
                ablated_norm = new_norm_upon_ablation_calc(clean_tokens, corrupted_tokens, ablation_type, heads = [(layer, head)], clean_cache = cache)
                norm_ratios[batch, ABLATION_TYPES.index(ablation_type), :, :] = (ablated_norm / clean_norm).cpu()
                
            
        # make histogram 
        new_norms = [norm_ratios[:, i, ...].flatten() for i in range(len(ABLATION_TYPES))]
        hist_data = [go.Histogram(x=norm, name=ablation_type, opacity=0.7) for ablation_type, norm in zip(ABLATION_TYPES, new_norms)]

        layout = go.Layout(
            title=f'Norm of Final Residual Stream when Ablation Layer {layer}, Head {head}',
            xaxis=dict(title='Ratio of Ablated to Clean Norm'),
            yaxis=dict(title='Frequency'),
            barmode='overlay'  # Overlay histograms for better comparison
        )

        fig = go.Figure(data=hist_data, layout=layout)
        fig.show()


# %%