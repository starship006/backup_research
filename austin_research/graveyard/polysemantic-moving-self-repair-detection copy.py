"""
polysemantic NMH behaviors! we know this to be the case in gpt-2 small so we can use this more generally and figure out
if we can classify head behaviors differently from this.
"""

# %%
from imports import *
import argparse
from reused_hooks import zero_ablation_hook
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
from scipy.stats import linregress
import torch.distributed as dist

from different_nmh_dataset_gen import generate_dataset
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
model.set_use_attn_result(True)
# %%
TEST_PROMPTS_PER_TYPE = 60 # called test cause its only the size appropriate for caching
PROMPTS, CORRUPTED_PROMPTS, ANSWERS, INCORRECT_ANSWERS, type_names = generate_dataset(model, TEST_PROMPTS_PER_TYPE * 4)

clean_tokens = model.to_tokens(PROMPTS)
corrupted_tokens = model.to_tokens(CORRUPTED_PROMPTS)

answer_tokens = model.to_tokens(ANSWERS, prepend_bos=False)
incorrect_answer_tokens = model.to_tokens(INCORRECT_ANSWERS, prepend_bos=False)

assert clean_tokens.shape == corrupted_tokens.shape
assert answer_tokens.shape == incorrect_answer_tokens.shape


SHORT_PROMPT_LEN = clean_tokens.shape[1]
# %%
combined_clean_tokens = torch.cat([clean_tokens, answer_tokens], dim=1)
combined_incorrect_tokens = torch.cat([corrupted_tokens, incorrect_answer_tokens], dim=1)

assert combined_clean_tokens.shape == combined_incorrect_tokens.shape
print("combined_clean_tokens MAY HAVE IMPROPERLY TOKENIZED STRINGS")
# %%
logits, clean_cache = model.run_with_cache(combined_clean_tokens)

# %% First, see which heads in the model are even useful for predicting these
from helpers import collect_direct_effect

per_head_direct_effect, all_layer_direct_effect = collect_direct_effect(clean_cache, correct_tokens=combined_clean_tokens,model = model,
                                                                        display=False)

# %%
imshow(per_head_direct_effect[..., -1].mean(-1), title = "average direct effect of each head on the last token across moving tasks")

top_heads = topk_of_Nd_tensor(per_head_direct_effect[..., -1].mean(-1), 3)

# %%
good_moving_layer = top_heads[0][0]
good_moving_head = top_heads[0][1]
# %%


@jaxtyped
@typechecker
def generate_data_vectors(layer, head, model = model, device = device, new_prompt_per_type_amount = TEST_PROMPTS_PER_TYPE):
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
    
    if new_prompt_per_type_amount != TEST_PROMPTS_PER_TYPE:
        PROMPTS, _, _, _, _ = generate_dataset(model, new_prompt_per_type_amount * 4)
        clean_tokens = model.to_tokens(PROMPTS)
    
    model.run_with_hooks(clean_tokens)
    model.reset_hooks()
    head_output_vectors = storage[0][:, -1, head, :] # get the outputs of an attentnion head at the last position
    
    
    flattened_outputs = head_output_vectors # no flattening this time
    flattened_labels = torch.zeros((new_prompt_per_type_amount * 4)).to(device)
    for i in range(4):
        flattened_labels[i * new_prompt_per_type_amount: (i+1) * new_prompt_per_type_amount] = i
    

    
    assert flattened_outputs.shape[0] == flattened_labels.shape[0]
    assert flattened_outputs.shape[1] == model.cfg.d_model
    assert len(flattened_outputs.shape) == 2
    assert len(flattened_labels.shape) == 1
    print(flattened_outputs.shape, flattened_labels.shape)
    return flattened_outputs, flattened_labels



flattened_outputs, flattened_labels = generate_data_vectors(good_moving_layer, good_moving_head, new_prompt_per_type_amount=300)


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
        flattened_outputs, flattened_labels = generate_data_vectors(layer, head, new_prompt_per_type_amount=300)
        _, accuracy = train_classifier_on_ablation_sklearn(flattened_outputs, flattened_labels, model)
        detection_ability[layer, head] = accuracy
# %%
fig = imshow(detection_ability, title = "classification accuracy of polymorphic activities in gpt2 small", return_fig=True)
# save to html
fig.write_html(f"clustering_results/{safe_model_name}-polymorphism-detection.html")
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
