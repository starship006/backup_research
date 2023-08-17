# %%
"""Neel Experiment:
1) Get a bunch of IOI prompts; ABA, ABB, BAA, BAB
2) Run the IOI task on them
3) For each of these, grab the src_resid - residual of the first name in resid_pre_10
4) Also grab the output of a Name Mover Head, such as 9.9
5) Take a backup head's QK circuit, and calculate  head_out @ W_Q @ W_K @ src_resid
"""

# !sudo apt install unzip
# !pip install git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python
# !pip install git+https://github.com/neelnanda-io/neel-plotly.git
# !pip install 
#!pip install plotly fancy_einsum jaxtyping transformers datasets transformer_lens
from imports import *
from different_nmh_dataset_gen import generate_dataset, generate_four_IOI_types
import unittest
# %%
model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(
    model_name,
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
    device = device,
)
model.set_use_attn_result(True)
# %% 
torch.cuda.empty_cache()
NUM_PROMPT_PER_TYPE = 400
NUM_PROMPTS = NUM_PROMPT_PER_TYPE * 4
ABB, ABA, BAA, BAB, NAMES = generate_four_IOI_types(model, NUM_PROMPTS // 4)
PROMPTS = ABB + ABA + BAA + BAB
INDEX_FIRST_NAME = 2
INDEX_SECOND_NAME = 4
PROMPT_TYPES = ["ABB", "ABA", "BAA", "BAB"]
IS_IO_FIRST_OR_SECOND = [0, 1, 0, 1]
# %%
clean_tokens = model.to_tokens(PROMPTS)
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
# %%

def get_effective_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:
    # from callum + arthur
    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)
    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid_pre = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(resid_pre)
    attn_out = einops.einsum(
        pre_attention, 
        model.W_V[0],
        model.W_O[0],
        "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
    )
    resid_mid = attn_out + resid_pre
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    W_ONLY_MLP = resid_pre.squeeze() + model.blocks[0].mlp(model.blocks[0].ln2(resid_pre)).squeeze()

    torch.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_U": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
        "Cody MLP": W_ONLY_MLP,
    }

# %%

def qk_composition_score(
        query_side: Float[Tensor, "d_model"],
        key_side: Float[Tensor, "d_model"],
        backup_head: Tuple
):
    W_Q = model.W_Q[backup_head[0], backup_head[1]]
    W_K = model.W_K[backup_head[0], backup_head[1]]
    return einops.repeat(query_side, "d_model -> 1 d_model") @ W_Q @ W_K.T @ key_side    

# %%
# attention scores on distribution
W_E = get_effective_embedding(model)["W_E (no MLPs)"]
name_mover_head = (9, 6)
self_repair_head = (10, 0)
layer_key = 10

diff_key_side_resid_streams = torch.zeros(4, clean_tokens.shape[-1])

ln_before_sr_head = model.blocks[self_repair_head[0]].ln1
resid_pre_before_sr_head = clean_cache[utils.get_act_name("resid_pre", self_repair_head[0])]

for prompt_type in range(4):
    for pos in range(clean_tokens.shape[-1]):
        score = 0
        for batch in range(NUM_PROMPT_PER_TYPE):            
            prompt = batch + prompt_type * NUM_PROMPT_PER_TYPE
            nmh_output = clean_cache[utils.get_act_name("result", name_mover_head[0])][prompt, -1, name_mover_head[1]]

            key_side = ln_before_sr_head(resid_pre_before_sr_head[prompt, pos])
            query_side =  ln_before_sr_head(nmh_output)
            score += qk_composition_score(query_side, key_side, self_repair_head).item()
        diff_key_side_resid_streams[prompt_type, pos] = score / NUM_PROMPT_PER_TYPE

imshow(diff_key_side_resid_streams,
       x = [(str(i) + "_" + j) for i,j in enumerate(model.to_str_tokens(clean_tokens[1]))],
       y=PROMPT_TYPES,
       title = f"query = output of 9.9 in distribution key = position attention in {self_repair_head}")
# %% 

def get_projection(from_vector, to_vector):
    if from_vector.shape == to_vector.shape and len(from_vector.shape) == 2:
        dot_product = einops.einsum(from_vector, to_vector, "batch d_model, batch d_model -> batch")
        squared_length_of_vector = einops.einsum(to_vector, to_vector, "batch d_model, batch d_model -> batch")
        projected_lengths = (dot_product) / (squared_length_of_vector)
        projections = to_vector * einops.repeat(projected_lengths, "batch -> batch d_model", d_model = to_vector.shape[-1])
    elif from_vector.shape == to_vector.shape and len(from_vector.shape) == 1:
        dot_product = einops.einsum(from_vector, to_vector, " d_model,  d_model -> ")
        squared_length_of_vector = einops.einsum(to_vector, to_vector, " d_model,  d_model -> ")
        projected_lengths = (dot_product) / (squared_length_of_vector)
        projections = to_vector * einops.repeat(projected_lengths, " ->  d_model", d_model = to_vector.shape[-1])
    else:
        print("BAD")
        return None
    return projections

def TestGetProjection():

    
    v1 = torch.tensor([1., 0., 0.]) 
    v2 = torch.tensor([0., 1., 0.])
    proj = get_projection(v1, v2)
    expected = torch.tensor([0., 0., 0.])
    assert torch.allclose(proj, expected)


    v1 = torch.tensor([1., 2., 3.])
    v2 = torch.tensor([2., 4., 6.]) 
    proj = get_projection(v1, v2)
    expected = v1
    assert torch.allclose(proj, expected)


    v1 = torch.tensor([1., 0., 0.])
    v2 = torch.tensor([1., 1., 0.]) 
    proj = get_projection(v1, v2)
    expected = torch.tensor([.5, .5, 0.])
    assert torch.allclose(proj, expected)

    v1 = torch.tensor([4,5,1])
    v2 = torch.tensor([2,3,4]) 
    proj = get_projection(v1, v2)
    expected = torch.tensor([1.862068965517241,2.793103448275862,3.724137931034483])
    assert torch.allclose(proj, expected)

if __name__ == '__main__':
    TestGetProjection()

# %% attention scores when projecting onto IO token
projected_diff_key_side_resid_streams = torch.zeros(4, clean_tokens.shape[-1])
for prompt_type in range(4):
    for pos in range(clean_tokens.shape[-1]):
        score = 0
        for batch in range(NUM_PROMPT_PER_TYPE):
            prompt = batch + prompt_type * NUM_PROMPT_PER_TYPE
            nmh_output = clean_cache[utils.get_act_name("result", name_mover_head[0])][prompt, -1, name_mover_head[1]]
        
            io_index = INDEX_FIRST_NAME if IS_IO_FIRST_OR_SECOND[prompt_type] == 0 else INDEX_SECOND_NAME
            unembedding = W_E[clean_tokens[prompt, io_index]] #- W_E[clean_tokens[prompt, index]]
            projection_to = get_projection(nmh_output, unembedding)

            key_side = ln_before_sr_head(resid_pre_before_sr_head[prompt, pos])
            query_side =  ln_before_sr_head(projection_to)
            score += qk_composition_score(query_side, key_side, self_repair_head).item()


        projected_diff_key_side_resid_streams[prompt_type, pos] = score / NUM_PROMPT_PER_TYPE

imshow(projected_diff_key_side_resid_streams,
       x = [(str(i) + "_" + j) for i,j in enumerate(model.to_str_tokens(clean_tokens[1]))],
       y=PROMPT_TYPES,
       title = f"query = output 9.9 from distributions, projected onto IO  |  key = position | attention in {self_repair_head}")
# %% attention scores when projecting onto S token
projected_diff_key_side_resid_streams = torch.zeros(4, clean_tokens.shape[-1])
for prompt_type in range(4):
    for pos in range(clean_tokens.shape[-1]):
        score = 0
        for batch in range(NUM_PROMPT_PER_TYPE):
            prompt = batch + prompt_type * NUM_PROMPT_PER_TYPE
            nmh_output = clean_cache[utils.get_act_name("result", name_mover_head[0])][prompt, -1, name_mover_head[1]]
        
            s_index = INDEX_SECOND_NAME if IS_IO_FIRST_OR_SECOND[prompt_type] == 0 else INDEX_FIRST_NAME

            unembedding = W_E[clean_tokens[prompt, s_index]] 
            projection_to = get_projection(nmh_output, unembedding)
            key_side = ln_before_sr_head(resid_pre_before_sr_head[prompt, pos])
            query_side =  ln_before_sr_head(projection_to)
            score += qk_composition_score(query_side, key_side, self_repair_head).item()


        projected_diff_key_side_resid_streams[prompt_type, pos] = score / NUM_PROMPT_PER_TYPE

imshow(projected_diff_key_side_resid_streams,
       x = [(str(i) + "_" + j) for i,j in enumerate(model.to_str_tokens(clean_tokens[1]))],
       y=PROMPT_TYPES,
       title = f"query = output 9.9 from distributions, projected onto S |  key = position | attention in {self_repair_head}")

# %% project onto logit diff
projected_diff_key_side_resid_streams = torch.zeros(4, clean_tokens.shape[-1])
for prompt_type in range(4):
    for pos in range(clean_tokens.shape[-1]):
        score = 0
        for batch in range(NUM_PROMPT_PER_TYPE):
            prompt = batch + prompt_type * NUM_PROMPT_PER_TYPE
            nmh_output = clean_cache[utils.get_act_name("result", name_mover_head[0])][prompt, -1, name_mover_head[1]]
        
            io_index = INDEX_FIRST_NAME if IS_IO_FIRST_OR_SECOND[prompt_type] == 0 else INDEX_SECOND_NAME
            s_index = INDEX_SECOND_NAME if IS_IO_FIRST_OR_SECOND[prompt_type] == 0 else INDEX_FIRST_NAME

            unembedding = W_E[clean_tokens[prompt, io_index]] - W_E[clean_tokens[prompt, s_index]]
            projection_to = get_projection(nmh_output, unembedding)

            key_side = ln_before_sr_head(resid_pre_before_sr_head[prompt, pos])
            query_side =  ln_before_sr_head(projection_to)
            score += qk_composition_score(query_side, key_side, self_repair_head).item()


        projected_diff_key_side_resid_streams[prompt_type, pos] = score / NUM_PROMPT_PER_TYPE

imshow(projected_diff_key_side_resid_streams,
       x = [(str(i) + "_" + j) for i,j in enumerate(model.to_str_tokens(clean_tokens[1]))],
       y=PROMPT_TYPES,
       title = f"query = output 9.9 from distributions, projected onto logit diff  |  key = position | attention in {self_repair_head}")


# %% project away from logit diff
projected_diff_key_side_resid_streams = torch.zeros(4, clean_tokens.shape[-1])
for prompt_type in range(4):
    for pos in range(clean_tokens.shape[-1]):
        score = 0
        for batch in range(NUM_PROMPT_PER_TYPE):
            prompt = batch + prompt_type * NUM_PROMPT_PER_TYPE
            nmh_output = clean_cache[utils.get_act_name("result", name_mover_head[0])][prompt, -1, name_mover_head[1]]
        
            io_index = INDEX_FIRST_NAME if IS_IO_FIRST_OR_SECOND[prompt_type] == 0 else INDEX_SECOND_NAME
            s_index = INDEX_SECOND_NAME if IS_IO_FIRST_OR_SECOND[prompt_type] == 0 else INDEX_FIRST_NAME

            unembedding = W_E[clean_tokens[prompt, io_index]] - W_E[clean_tokens[prompt, s_index]]
            projection_to = get_projection(nmh_output, unembedding)

            key_side = ln_before_sr_head(resid_pre_before_sr_head[prompt, pos])
            query_side =  ln_before_sr_head(nmh_output - projection_to)
            score += qk_composition_score(query_side, key_side, self_repair_head).item()


        projected_diff_key_side_resid_streams[prompt_type, pos] = score / NUM_PROMPT_PER_TYPE

imshow(projected_diff_key_side_resid_streams,
       x = [(str(i) + "_" + j) for i,j in enumerate(model.to_str_tokens(clean_tokens[1]))],
       y=PROMPT_TYPES,
       title = f"query = output 9.9 from distributions, projected away from logit diff  |  key = position | attention in {self_repair_head}")








# %% see if this is a positional thing or a positional + token unembedding thing

results = torch.zeros(clean_tokens.shape[-1], 4)

for prompt in range(NUM_PROMPT_PER_TYPE):
    nmh_output = clean_cache[utils.get_act_name("result", name_mover_head[0])][prompt, -1, name_mover_head[1]]

    for pos in range(clean_tokens.shape[-1]):
        score = 0
        resid_pre_before_sr_head = clean_cache[utils.get_act_name("resid_pre", 10)]

        for key_prompt in range(4):
            key_side = ln_before_sr_head(resid_pre_before_sr_head[prompt + NUM_PROMPT_PER_TYPE * key_prompt, pos])
            query_side =  ln_before_sr_head(nmh_output)
            results[pos, key_prompt] += qk_composition_score(query_side, key_side, self_repair_head).item()
        


# %%
imshow(results.T, x = [(str(i) + "_" + j) for i,j in enumerate(model.to_str_tokens(clean_tokens[1]))], y=PROMPT_TYPES, title = f"query = output of 9.9 from ABB, key = resid from different distributions, head = {self_repair_head}")
# %%
