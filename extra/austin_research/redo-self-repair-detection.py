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
in_notebook_mode = False
if in_notebook_mode:
    model_name = "pythia-160m"
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
                        device: str, layer_to_ablate: int = 9, 
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
def train_classifier_on_ablation_sklearn(model, flattened_resid, flattened_labels):
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
# train_classifier_on_ablation_sklearn(model)
# %%
from sklearn.cluster import KMeans
@typechecker
def cluster_vectors(vectors, num_clusters: int = 8, verbose: bool = True):
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
