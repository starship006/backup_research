"""
polysemantic NMH behaviors! we know this to be the case in gpt-2 small so we can use this more generally and figure out
if we can classify head behaviors differently from this.


this time we use the new dataset generation function, which should hopefully cover enough invariants to get a good measure
about whether or not we are genuinely capturing something polysemantic
"""

# %%
from imports import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import argparse
from reused_hooks import zero_ablation_hook
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
from scipy.stats import linregress
import torch.distributed as dist

from updated_nmh_dataset_gen import generate_ioi_mr_prompts
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
    model_name = "pythia-160m" # PROMPT CHECKER ONLY CHECKED FOR GPT2 TOKENIZER; NEED TO UPDATE DATASET FUNC FOR NON-GPT2 MDOELS
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
# to avoid accidentally calling this TEST_PROMPTS_PER_TYPE =  # called test cause its only the size appropriate for caching
PROMPTS, ANSWERS, ANSWER_INDICIES = generate_ioi_mr_prompts(model, 60)

# %%
clean_tokens: Float[Tensor, "batch pos"] = model.to_tokens(PROMPTS).to(device)
answer_tokens: Float[Tensor, "batch 1"] = model.to_tokens(ANSWERS, prepend_bos=False).to(device)
answer_token_idx: Float[Tensor, "batch"] = torch.tensor(ANSWER_INDICIES).to(device)

# %%
# See if the answer_token_idx is correct
unsqueezed_answers_idx = answer_token_idx.unsqueeze(-1)
indexed_answers = clean_tokens.gather(-1, unsqueezed_answers_idx)
assert torch.all(torch.eq(answer_tokens, indexed_answers))
# %%
logits, cache = model.run_with_cache(clean_tokens)

# %% First, see which heads in the model are even useful for predicting these
from helpers import collect_direct_effect

per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(cache, correct_tokens=clean_tokens,model = model,
                                                                        display=False)

# %% Get direct effect for each batch
expanded_indicies = einops.repeat(answer_token_idx, "b -> a c b 1", a = model.cfg.n_layers, c = model.cfg.n_layers)

# %%
important_direct_effect: Float[Tensor, "layer head batch"] = per_head_direct_effect.gather(-1, expanded_indicies - 1).squeeze()
# %%
imshow(important_direct_effect[..., 0:(int(important_direct_effect.shape[-1] / 2))].mean(-1), title = "average de of head on IOI prompts")
imshow(important_direct_effect[..., (int(important_direct_effect.shape[-1] / 2)):].mean(-1), title = "average de of head on MR/MRS prompts")
# %%
top_heads = topk_of_Nd_tensor(important_direct_effect[..., (int(important_direct_effect.shape[-1] / 2)):].mean(-1), 3)

# %%
good_moving_layer = top_heads[0][0]
good_moving_head = top_heads[0][1]
# %%
# CREATE GLOBAL PROMPTS FOR THE FOLLOWING TRAINING
global_prompts_per_type = 1000
global_prompts, _, global_answer_idx = generate_ioi_mr_prompts(model, global_prompts_per_type)

# %%

@jaxtyped
@typechecker
def generate_data_vectors(layer, head, model = model, device = device, prompts_per_type = 1000) -> Tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch"]]:
    """
    gets output vectors for a head and sees creates labels for it depending on the four types
    """
    # Pick random indices for owt_dataset tokens
    
    
    all_resids = []
    all_labels = []

    def store_item_hook(
        item,
        hook,
        where_to_store: list
    ):
        where_to_store.append(item.clone())


    model.reset_hooks()
    storage = []
    model.add_hook(utils.get_act_name("result", layer), partial(store_item_hook, where_to_store = storage))
    
    if prompts_per_type == global_prompts_per_type:
        prompts = global_prompts
        answer_idx = global_answer_idx
    else:
        prompts, _, answer_idx = generate_ioi_mr_prompts(model, prompts_per_type)    


    local_clean_tokens: Float[Tensor, "batch pos"] = model.to_tokens(prompts).to(device)
    local_answer_token_idx: Float[Tensor, "batch"] = torch.tensor(answer_idx).to(device)

    model.run_with_hooks(local_clean_tokens)
    model.reset_hooks()
    
    head_output_vectors: Float[Tensor, "batch pos d_model"] = storage[0][..., head, :] # get the outputs of an attentnion head 
    expanded_indicies = einops.repeat(local_answer_token_idx, "b -> b 1 c", c = model.cfg.d_model)
    
    flattened_outputs: Float[Tensor, "batch d_model"] = head_output_vectors.gather(1, expanded_indicies - 1).squeeze()
    flattened_labels = torch.zeros(flattened_outputs.shape[0])
    flattened_labels[(prompts_per_type * 2):] = 1
    

    # Quick test to ensure we gathered correct tensors
    assert flattened_outputs[0].eq(head_output_vectors[0, answer_idx[0] - 1, :]).all()
    assert flattened_outputs[1].eq(head_output_vectors[1, answer_idx[1] - 1, :]).all()
    assert flattened_outputs[2].eq(head_output_vectors[2, answer_idx[2] - 1, :]).all()

    # Test to ensure output and labels are the right size
    assert flattened_outputs.shape[0] == flattened_labels.shape[0]
    assert flattened_outputs.shape[1] == model.cfg.d_model
    assert len(flattened_outputs.shape) == 2
    assert len(flattened_labels.shape) == 1
    print(flattened_outputs.shape, flattened_labels.shape)
    return flattened_outputs, flattened_labels



flattened_outputs, flattened_labels = generate_data_vectors(good_moving_layer, good_moving_head, prompts_per_type=1000)


# %%
def preprocess_data(flattened_outputs, flattened_labels):
    # Convert Tensors to numpy arrays if necessary
    if isinstance(flattened_outputs, torch.Tensor):
        flattened_outputs = flattened_outputs.cpu().numpy()
    if isinstance(flattened_labels, torch.Tensor):
        flattened_labels = flattened_labels.cpu().numpy()

    # Skip filtering for balance - dataset generation guarantees they all have the same amount


    # Shuffle the data
    shuffled_indices = np.random.permutation(len(flattened_outputs))
    shuffled_outputs = flattened_outputs[shuffled_indices]
    shuffled_labels = flattened_labels[shuffled_indices]



    # Split into training and testing sets
    split_idx = len(shuffled_outputs) * 2 // 3
    train_outputs = shuffled_outputs[:split_idx]
    test_outputs = shuffled_outputs[split_idx:]
    train_labels = shuffled_labels[:split_idx]
    test_labels = shuffled_labels[split_idx:]
    return train_outputs, test_outputs, train_labels, test_labels


def train_classifier_on_ablation_sklearn(flattened_outputs, flattened_labels, model = model):
    
    train_outputs, test_outputs, train_labels, test_labels = preprocess_data(flattened_outputs, flattened_labels)
    
    if train_outputs.shape[0] <= 100:
        print("not enough data")
        return None, 0


    # Train the classifier
    classifier = sklearn.linear_model.LogisticRegression()  # or any other scikit-learn classifier
    classifier.fit(train_outputs, train_labels)
    
    # Predict and calculate accuracy on training set
    predicted = classifier.predict(train_outputs)
    accuracy = accuracy_score(train_labels, predicted)

    # Get Test Accuracy
    test_predicted = classifier.predict(test_outputs)
    test_accuracy = accuracy_score(test_labels, test_predicted)
    
    print(f"Training Accuracy: {accuracy:.5f}", f"Test Accuracy: {test_accuracy:.5f}")
    print("Classification Report:\n", classification_report(test_labels, test_predicted))
    return classifier, test_accuracy  # return the trained model

# %%
train_classifier_on_ablation_sklearn(flattened_outputs, flattened_labels)




# %%
detection_ability = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

# %%
for layer in range(0, model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        print(f"layer {layer} head {head}")
        flattened_outputs, flattened_labels = generate_data_vectors(layer, head, prompts_per_type=500)
        _, accuracy = train_classifier_on_ablation_sklearn(flattened_outputs, flattened_labels, model)
        detection_ability[layer, head] = accuracy
# %%
fig = imshow(detection_ability, title = "classification accuracy of polymorphic activities in gpt2 small", return_fig=True)
# save to html
#fig.write_html(f"clustering_results/{safe_model_name}-polymorphism-detection.html") -- old data may be bad
fig.show()

# %%
from sklearn.cluster import KMeans
@typechecker
def cluster_vectors(vectors, num_clusters, verbose: bool = True):
    """
    Clusters the vectors using the KMeans algorithm, providing runtime statistics and visualization.
    
    Args:
        vectors (Tensor): The vectors to be clustered.
        num_clusters (int): The number of clusters to form.
        verbose (bool): Print runtime statistics if True.
    
    Returns:
        Tuple: A tuple containing the cluster labels and the KMeans model.
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=verbose)
    labels = kmeans.fit_predict(vectors.cpu().numpy())

    # Print final statistics
    if verbose:
        print(f"Final inertia: {kmeans.inertia_}")
        print(f"Number of iterations: {kmeans.n_iter_}")
    return labels, kmeans

# %%

# %%
from sklearn.decomposition import PCA
@typechecker
def visualize_and_evaluate(reduced_vectors, labels, true_labels, num_clusters: int):
    """
    Visualizes the clustered vectors and calculates metrics.

    Args:
        vectors (Tensor): The vectors that were clustered.
        labels (ndarray): The cluster labels from the clustering algorithm.
        true_labels (Tensor): The ground truth labels.
        num_clusters (int): The number of clusters used.

    Returns:
        Figure: A Plotly figure for visualization.
    """

    # Process tensors into numpy
    if type(reduced_vectors) == torch.Tensor:
        reduced_vectors = reduced_vectors.cpu().numpy()
    if type(labels) == torch.Tensor:
        labels = labels.cpu().numpy()
    if type(true_labels) == torch.Tensor:
        true_labels = true_labels.cpu().numpy()

    if len(true_labels.shape) == 2:
        print("Flattening true labels")
        true_labels = np.argmax(true_labels, axis=1)
    if len(labels.shape) == 2:
        print("Flattening true labels")
        labels = np.argmax(labels, axis=1)
    

    # Create a Plotly scatter plot
    if reduced_vectors.shape[1] == 2:
        fig = px.scatter(reduced_vectors, x=0, y=1, color=labels, title='Cluster Visualization', size_max=0.2)
    elif reduced_vectors.shape[1] == 3:
        fig = px.scatter_3d(reduced_vectors, x=0, y=1, z=2, color=labels, title='Cluster Visualization', size_max=0.01, opacity = 0.5)
        fig.update_traces(marker=dict(size=2))
    
    # Calculate metrics
    # silhouette_avg = silhouette_score(vectors.cpu().numpy(), labels)
    # calinski_harabasz = calinski_harabasz_score(vectors.cpu().numpy(), labels)
    # davies_bouldin = davies_bouldin_score(vectors.cpu().numpy(), labels)

    # print(f"Silhouette Coefficient: {silhouette_avg:.2f}")
    # print(f"Calinski-Harabasz Index: {calinski_harabasz:.2f}")
    # print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
    stats = {}
    stats['Rand Index'] = sklearn.metrics.adjusted_rand_score(true_labels, labels)
    stats['Adjusted Mutual Information Score'] = sklearn.metrics.adjusted_mutual_info_score(true_labels, labels)
    stats['Homogeneity Score'] = sklearn.metrics.homogeneity_score(true_labels, labels)
    stats['Completeness Score'] = sklearn.metrics.completeness_score(true_labels, labels)

    # Printing the results
    for key, value in stats.items():
        print(f"{key}: {value}")



    return fig, stats


# %% Reduced the vectors
stats_storage = []
for layer in range(model.cfg.n_heads):
    print(f"Here are stats on layer {layer}")
    flattened_resids, flattened_labels = generate_vectors(model, True, 100, model.cfg.n_heads, 40, device, 
                                                                        layer_to_ablate=layer, act_to_read = utils.get_act_name("resid_post", layer), dataset = dataset)
    
    num_clusters = model.cfg.n_heads  # Adjust as needed
    cluster_labels, kmeans_model = cluster_vectors(flattened_resids, num_clusters=num_clusters, verbose=True)
    
    pca = PCA(n_components=3)
    reduced_vectors = pca.fit_transform(flattened_resids.cpu().numpy())
    
    # Visualization with Plotly
    fig, stats_for_layer = visualize_and_evaluate(reduced_vectors, cluster_labels, flattened_labels, num_clusters=num_clusters)
    fig.update_layout(title=f"PCA Visualization of {model_name} Residual Vectors")
    fig.show()

    stats_storage.append(stats_for_layer)

# %%


def plot_combined_metrics(stats_storage):
    # Initialize dictionaries to store metric values for each layer
    metric_values = {
        'Rand Index': [],
        'Adjusted Mutual Information Score': [],
        'Homogeneity Score': [],
        'Completeness Score': []
    }

    # Accumulate metric values for each layer
    for stats in stats_storage:
        for metric in metric_values.keys():
            metric_values[metric].append(stats[metric])

    # Create a single Plotly figure with multiple traces
    fig = go.Figure()
    for metric, values in metric_values.items():
        fig.add_trace(go.Scatter(
            y=values,
            mode='lines+markers',
            name=metric
        ))

    # Update layout of the figure
    fig.update_layout(
        title="Metrics Across Layers",
        xaxis=dict(title='Layer'),
        yaxis=dict(title='Metric Values'),
        legend_title="Metrics"
    )
    
    if in_notebook_mode:
        fig.show()
    else:
        # save figure to html
        fig.write_html(f"clustering_results/{safe_model_name}-metrics.html")

# Usage
plot_combined_metrics(stats_storage)

# %%
