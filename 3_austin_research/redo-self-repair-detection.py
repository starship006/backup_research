"""
okay im going to use good coding practices for dataset generation to actually use sckit learn
to learn stuff. lets see what happens
"""

# %%
from imports import *
import argparse
from reused_hooks import zero_ablation_hook
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
from scipy.stats import linregress
import torch.distributed as dist
# %%
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from beartype import beartype as typechecker
from sklearn.neighbors import KNeighborsClassifier
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
#model.set_use_attn_result(True)
# %%
dataset = utils.get_dataset("owt")
dataset_name = "owt"
torch.set_grad_enabled(False)
# %%

@jaxtyped
@typechecker
def generate_vectors(model, isolated: bool,
                    BATCH_SIZE: int, num_heads: int, PROMPT_LEN: int, 
                        device: str, layer_to_ablate: int =9, 
                        act_to_read:str = utils.get_act_name("resid_mid", 9), dataset: datasets.arrow_dataset.Dataset = dataset):
    """
    if isolated:
        vector = original vector -  ablated residual vector
    else:
        vector = ablated residual vector
    picks random prompts for each epoch BY ZERO ABLATING
    each ablated head has PROMPT_LEN * BATCH_SIZE residual stream vectors generated for it
    """
    # Pick random indices for owt_dataset tokens
    random_indices = random.sample(range(len(dataset)), BATCH_SIZE)
    all_owt_tokens = model.to_tokens([dataset[i]["text"] for i in random_indices]).to(device)
    
    all_resids = []
    all_labels = []

    def store_item_hook(
        resid_in_model,
        hook,
        where_to_store: list
    ):
        where_to_store.append(resid_in_model.clone())


    def space_efficient_dir_effects_from_zero_ablating(model, owt_tokens, layer_to_ablate, head_to_ablate, act_to_read = act_to_read):
        """
        zero ablates an attention head and returns a specific activation
        """

        model.reset_hooks()
        model.add_hook(utils.get_act_name("z", layer_to_ablate), partial(zero_ablation_hook, head_index_to_ablate=head_to_ablate))
        stored_acts = []
        model.add_hook(act_to_read, partial(store_item_hook, where_to_store=stored_acts))
        model.run_with_hooks(owt_tokens)
        model.reset_hooks()
        return stored_acts[0]


    for head_to_ablate in range(num_heads):
        # generate the residual stream vectors
        owt_tokens = all_owt_tokens[..., 0:PROMPT_LEN]
        
        resid_pre = space_efficient_dir_effects_from_zero_ablating(model, owt_tokens, layer_to_ablate, head_to_ablate)
        
        if isolated:
            # get the original vectors
            _, clean_cache = model.run_with_cache(owt_tokens)
            clean_resid_pre = clean_cache[act_to_read]

            # subtract the residual vectors and add to list
            all_resids.append(clean_resid_pre - resid_pre)
        else:
            all_resids.append(resid_pre)

        # generate labels
        labels = F.one_hot(torch.tensor([head_to_ablate]), num_classes=num_heads).to(device).float()
        all_labels.append(einops.repeat(labels, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))
        
    concatenated_resids = torch.cat(all_resids, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)
    

    
    flattened_resids = einops.rearrange(concatenated_resids, "b p h -> (b p) h")
    flattened_labels = einops.rearrange(concatenated_labels, "b p h -> (b p) h")
    

    
    assert flattened_resids.shape[0] == flattened_labels.shape[0]
    assert flattened_resids.shape[1] == model.cfg.d_model
    assert len(flattened_resids.shape) == 2
    
    return flattened_resids, flattened_labels




# %%

flattened_resids, flattened_labels = generate_vectors(model, False, 300, model.cfg.n_heads, 80, device, 
                                                                       layer_to_ablate=9, act_to_read = utils.get_act_name("resid_post", 9), dataset = dataset)
        

# %%
def train_classifier_on_ablation_sklearn(model, flattened_resid = flattened_resids, flattened_labels = flattened_labels):
    classifier = sklearn.svm.LinearSVC()  # or any other scikit-learn classifier
    best_accuracy = 0
    
    # Reshape labels for scikit-learn
    labels = np.argmax(flattened_labels.cpu().numpy(), axis=1)
    
    
    # Train the classifier
    classifier.fit(flattened_resids.cpu().numpy(), labels)
    
    # Predict and calculate accuracy on training set
    predicted = classifier.predict(flattened_resids.cpu().numpy())
    accuracy = accuracy_score(labels, predicted)

    # Get Test Accuracy
    test_resids, test_labels = generate_vectors(model, True, 20, model.cfg.n_heads, 80, device, 
                                                                       layer_to_ablate=9, act_to_read = utils.get_act_name("resid_post", 9), dataset = dataset)
    
    test_labels = np.argmax(test_labels.cpu().numpy(), axis=1)
    test_predicted = classifier.predict(test_resids.cpu().numpy())
    test_accuracy = accuracy_score(test_labels, test_predicted)
    
    print(f"Training Accuracy: {accuracy:.5f}", f"Test Accuracy: {test_accuracy:.5f}")
    print("Classification Report:\n", classification_report(test_labels, test_predicted))
    return classifier  # return the trained model

# %%
train_classifier_on_ablation_sklearn(model)
# %%
