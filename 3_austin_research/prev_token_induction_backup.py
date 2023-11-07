# %%
""" Motivation; We want to explore the consequences of ablating previous token heads vs induction heads on
how much induction the model displays.

If prev token ablation doesn't break induction, this is potentially interesting
"""
from imports import *
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
# %%
in_notebook_mode = True
if in_notebook_mode:
    model_name = "gpt2-small"
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
# %% Code to generate induction tokens

def generate_clean_corrupted_induction_prompts(model, gen_size = 600, prompt_half_length = 10):
    """
    generates a series of clean and corrupted random token induction prompts
    answer tokens are associated with the correct token in the clean prompt, as well as the correct token in the corrupted prompt
    """
    def generate_repeated_tokens(model, batch, seq_len) -> Float[Tensor, "batch seq_len*2"]:
        tokens = torch.randint(1, model.cfg.d_vocab, (batch, seq_len))
        return torch.concat((tokens, tokens), dim=-1)
    
    gen_tokens = generate_repeated_tokens(model, gen_size, prompt_half_length).cuda()
    gen_prompts = [''.join(model.to_str_tokens(gen_tokens[i, :])) for i in range(gen_tokens.shape[0])]

    induction_tokens = []
    induction_prompts = []

    for i in range(gen_size):
    # for prompts that have the same tokens before and after, add to prompts
        if gen_tokens[i].equal(model.to_tokens(gen_prompts[i], prepend_bos=False)[0]):
            induction_tokens.append(gen_tokens[i])
        #induction_prompts.append(gen_prompts[i])

    for i in range(len(induction_tokens)):
    #print(good_tokens[i])
    #print(model.to_tokens(model.to_string(good_tokens[i]), prepend_bos=False))
        assert induction_tokens[i].equal(model.to_tokens(model.to_string(induction_tokens[i]), prepend_bos=False)[0])
    #assert induction_prompts[i] == model.to_string(induction_tokens[i])


    for i, prompt in enumerate(induction_tokens):
    # remove last token
        induction_tokens[i] = prompt[:-1]
    # generate new prompt and add to induction prompts
        induction_prompts.append(model.to_string(induction_tokens[i]))

    broken_batch_size = len(induction_tokens)
    possible_broken_corrupted_tokens = []
    possible_broken_corrupted_prompts = []

    for i in range(broken_batch_size):
        temp = induction_tokens[i].clone()
        temp[prompt_half_length] = torch.randint(1, model.cfg.d_vocab, (1,))
        possible_broken_corrupted_tokens.append(temp)
        possible_broken_corrupted_prompts.append(model.to_string(temp))
    
    BOS_TOKEN = model.to_tokens("")[0].item()

    # filter possible_broken_corrupted_tokens and possible_broken_corrupted_prompts to only include only the ones to which tokenization remains same
    corrupted_tokens = []
    corrupted_prompts = []

    # ALSO ADDS BOS!
    num_removed = 0
    for i in range(broken_batch_size):
        if possible_broken_corrupted_tokens[i].equal(model.to_tokens(possible_broken_corrupted_prompts[i], prepend_bos=False)[0]):
            new_corrupted_token = torch.cat((torch.tensor([BOS_TOKEN]).cuda(), possible_broken_corrupted_tokens[i])).cuda()
            corrupted_tokens.append(new_corrupted_token)
            corrupted_prompts.append(model.to_string(new_corrupted_token))

            new_clean_token = torch.cat((torch.tensor([BOS_TOKEN]).cuda(), induction_tokens[i - num_removed])).cuda()
            induction_tokens[i - num_removed] = new_clean_token
            induction_prompts[i - num_removed] = model.to_string(new_clean_token)
        else:
        # remove associated induction prompt
            induction_tokens.pop(i - num_removed)
            induction_prompts.pop(i - num_removed)
            num_removed += 1

    assert len(corrupted_tokens) == len(corrupted_prompts) == len(induction_tokens) == len(induction_prompts)
    batch_size = len(corrupted_tokens)
    clean_tokens = torch.stack(induction_tokens)
    corrupted_tokens = torch.stack(corrupted_tokens)
    answer_tokens = torch.stack((clean_tokens[:, prompt_half_length], corrupted_tokens[:, prompt_half_length]), dim=1)


    # test one - make sure that answer token only shows up once
    assert (clean_tokens[0] == answer_tokens[0, 0]).sum() == 1

    # delete temp variables
    del possible_broken_corrupted_tokens
    del possible_broken_corrupted_prompts
    return corrupted_tokens,batch_size,clean_tokens,answer_tokens
# %% generate completely different corrupted prompts
_, batch_size_clean, clean_tokens, answer_tokens_clean = generate_clean_corrupted_induction_prompts(model, gen_size = 100, prompt_half_length=36)
_, batch_size_corrupted, corrupted_tokens, answer_tokens_corrupted = generate_clean_corrupted_induction_prompts(model, gen_size = 100, prompt_half_length=36)

# merge into single group
batch_size = min(batch_size_clean, batch_size_corrupted, 20)
clean_tokens = clean_tokens[:batch_size]
corrupted_tokens = corrupted_tokens[:batch_size]
answer_tokens = torch.stack((answer_tokens_clean[:batch_size, 0], answer_tokens_corrupted[:batch_size, 0]), dim=1)

# %%
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)
# %%
answer_residual_directions: Float[Tensor, "batch 2 d_model"] = model.tokens_to_residual_directions(answer_tokens)
correct_residual_direction, incorrect_residual_direction = answer_residual_directions.unbind(dim=1)
logit_diff_directions: Float[Tensor, "batch d_model"] = correct_residual_direction - incorrect_residual_direction
# %%
def logits_to_ave_logit_diff(
    logits: Float[Tensor, "batch seq d_vocab"],
    answer_tokens: Float[Tensor, "batch 2"] = answer_tokens,
    per_prompt: bool = False
):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''
    # Only the final logits are relevant for the answer
    final_logits: Float[Tensor, "batch d_vocab"] = logits[:, -1, :]
    # Get the logits corresponding to the indirect object / subject tokens respectively
    answer_logits: Float[Tensor, "batch 2"] = final_logits.gather(dim=-1, index=answer_tokens.to(device))
    # Find logit difference
    correct_logits, incorrect_logits = answer_logits.unbind(dim=-1)
    answer_logit_diff = correct_logits - incorrect_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()

def induction_attn_detector(cache: ActivationCache) -> Float[Tensor, "layer head"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) where each element is the average attention to induction token
    '''
    
    attn_heads = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][:, head].mean(0)
            # take avg of (-seq_len+1)-offset elements
            seq_len = (attention_pattern.shape[-1]) // 2
            #print(seq_len)
            attn_heads[layer, head] = attention_pattern.diagonal(-seq_len+1).mean()
            #if layer == 7:
                #print(f"layer {layer} head {head} attn to induction token: {attention_pattern.diagonal(-seq_len+1).shape}")
    return attn_heads

def prev_attn_detector(cache: ActivationCache) -> Float[Tensor, "layer head"]:
    '''
    Returns a tensor of shape (n_layers, n_heads) where each element is the average attention to previous token
    '''
    attn_heads = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            attention_pattern = cache["pattern", layer][:, head].mean(0)
            # take avg of sub-diagonal elements
            attn_heads[layer, head] = attention_pattern.diagonal(-1).mean()
    return attn_heads

def stare_at_attention_and_head_pat(cache, layer_to_stare_at, head_to_isolate, display_corrupted_text = False, verbose = True):
  """
  given a cache from a run, displays the attention patterns of a layer, as well as printing out how much the model
  """

  tokenized_str_tokens = model.to_str_tokens(corrupted_tokens[0]) if display_corrupted_text else model.to_str_tokens(clean_tokens[0])
  attention_patten = cache["pattern", layer_to_stare_at]
  print(f"Layer {layer_to_stare_at} Head {head_to_isolate} Activation Patterns:")

  if verbose:
    display(cv.attention.attention_heads(
      tokens=tokenized_str_tokens,
      attention=attention_patten.mean(0),
      #attention_head_names=[f"L{layer_to_stare_at}H{i}" for i in range(model.cfg.n_heads)],
    ))
  else:
    print(attention_patten.mean(0).shape)

    display(cv.attention.attention_patterns(
      tokens=tokenized_str_tokens,
      attention=attention_patten.mean(0),
      attention_head_names=[f"L{layer_to_stare_at} H{i}" for i in range(model.cfg.n_heads)],
    ))

# %%
attention_scores = prev_attn_detector(clean_cache)
imshow(attention_scores, title = "Average Attention to previous token")

# %%
induction_scores = induction_attn_detector(clean_cache)
imshow(induction_scores, title = "Average Attention to induction token")

# %%

# %% Experiment: path patch from one prev token head to downstream induction heads and see whats up
top_ptoken_heads = topk_of_Nd_tensor(attention_scores, 3)

#prev_token_layer = 4
#prev_token_head = 11
path_patch_results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        path_patch_results[layer,head] = path_patch(model,
        orig_input = clean_tokens,
        new_input = corrupted_tokens, 
        sender_nodes = [Node("z", prev_token_layer, prev_token_head) for prev_token_layer, prev_token_head in top_ptoken_heads],
        receiver_nodes = [Node("q", layer, head), Node("k", layer, head), Node("v", layer, head)],
        patching_metric = lambda x: induction_attn_detector(x)[layer, head],
        apply_metric_to_cache = True,
        verbose = False
        )

# %% Next, just activation patch
activation_patch_results = act_patch(model,
    orig_input = clean_tokens,
    new_input = corrupted_tokens, 
    patching_nodes = [Node("z", prev_token_layer, prev_token_head) for prev_token_layer, prev_token_head in top_ptoken_heads],
    patching_metric = induction_attn_detector,
    apply_metric_to_cache = True,
    verbose = False
    )

# %%


imshow(
    torch.stack([induction_scores, path_patch_results - induction_scores, activation_patch_results - induction_scores]),
    return_fig = True,
    facet_col = 0,
    facet_labels = ["Induction Scores", "Path Patch Diff", "Activation Patch Diff"],
    title=f"Induction Scores when Path Patching and Activation Patching from prev token head {top_ptoken_heads}",
    labels={"x": "Head", "y": "Layer", "color": "Avg Attn to Inducted Token"},
    #coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1000,
    margin={"r": 100, "l": 100}
)

# %% Experiment two: patch from induction heads to induction heads
top_induction_heads = topk_of_Nd_tensor(induction_scores, 3)
#induction_layer = 5
#induction_head = 5
induction_patch_results = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        induction_patch_results[layer,head] = path_patch(model,
        orig_input = clean_tokens,
        new_input = corrupted_tokens, 
        sender_nodes = [Node("z", induction_layer, induction_head) for induction_layer, induction_head in top_induction_heads],
        receiver_nodes = [Node("q", layer, head), Node("k", layer, head), Node("v", layer, head)],
        patching_metric = lambda x: induction_attn_detector(x)[layer, head],
        apply_metric_to_cache = True,
        verbose = False
        )
# %%
# NOTE! i've included extra 7.1 head inside here right now
induction_activation_patch_results = act_patch(model,
    orig_input = clean_tokens,
    new_input = corrupted_tokens, 
    patching_nodes = [Node("z", induction_layer, induction_head) for induction_layer, induction_head in top_induction_heads],
    patching_metric = induction_attn_detector,
    apply_metric_to_cache = True,
    verbose = False
    )

# %%

imshow(
    torch.stack([induction_scores, induction_patch_results - induction_scores, induction_activation_patch_results - induction_scores]),
    return_fig = True,
    facet_col = 0,
    facet_labels = ["Induction Scores", "Path Patch Diff", "Activation Patch Diff"],
    title=f"Induction Scores when Path Patching and Activation Patching from induction token head {top_induction_heads}",
    labels={"x": "Head", "y": "Layer", "color": "Avg Attn to Inducted Tokenn"},
    #coloraxis=dict(colorbar_ticksuffix = "%"),
    border=True,
    width=1000,
    margin={"r": 100, "l": 100}
)

# %%
