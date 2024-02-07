"""
okay im going to use good coding practices for dataset generation to actually use sckit learn
to learn stuff. 

we are now going to conduct a different test to see whether or not an attention head doing something else can be detected
by 
"""

# %%
from imports import *
import argparse
from reused_hooks import zero_ablation_hook
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
from scipy.stats import linregress
import torch.distributed as dist

from different_nmh_dataset_gen import generate_four_IOI_types
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
dataset = utils.get_dataset("owt")
dataset_name = "owt"
torch.set_grad_enabled(False)
# %% First explore whether or not heads attend to punctuation at variable amounts
BATCH_SIZE = 200
PROMPT_LEN = 80
clean_tokens = model.to_tokens([dataset[i]["text"] for i in range(BATCH_SIZE)]).to(device)[:, :PROMPT_LEN]

# %% Get indicies for punctuation
PUNCTUATION: List[int] = [i[0].item() for i in model.to_tokens(
        [".", ",", "!", "?", "(", ")", "-", "_", "=", "+", "[", "]", "{", "}", "|", ";", ":", "'", "\"", "/", "<", ">", "`", "~"], 
        prepend_bos=False
    )]
punctuation_indicies = []
for prompt in clean_tokens:
    punctuation_indicies.append([i for i, x in enumerate(prompt) if x in PUNCTUATION])      
    

# %%
@jaxtyped
@typechecker
def hook_percent_attn_punctuation(
    attn_pattern: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head_index: int,
    layer_index: int,
    tensor_to_write_into: Float[Tensor, "n_layers n_heads"],
    punctuation_indicies: List[List[int]] = [] # Should be a list that holds batch amount of lists of indicies for punctuation,
):
    head_attn = attn_pattern[:, head_index, :, :]
    max_token_attended_to: Float[Tensor, "batch seqQ"] = torch.argmax(head_attn, dim=-1)
    # for each batch, count how many seqQ are punctuation
    global_punc_count = 0
    for batch, p_indicies in enumerate(punctuation_indicies):
        token_max_attn_count = max_token_attended_to[batch].bincount(minlength = PROMPT_LEN)
        punc_count = 0
        for p_index in p_indicies:
            punc_count += token_max_attn_count[p_index]

        global_punc_count += punc_count

    tensor_to_write_into[layer_index, head_index] = global_punc_count / (BATCH_SIZE * PROMPT_LEN)
        
        

        


@jaxtyped
@typechecker
def get_head_percent_attend_to_punctuation(model = model, clean_tokens = clean_tokens, punctuation_indicies = punctuation_indicies):
    """
    runs the model on clean_tokens and gathers what percentage of times the head attends to punctuation
    """
    model.reset_hooks()
    

    result_tensor = torch.zeros((model.cfg.n_layers, model.cfg.n_heads)).to(device)
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            model.add_hook(utils.get_act_name("pattern", layer),
                            partial(hook_percent_attn_punctuation, head_index = head, layer_index = layer,
                                    tensor_to_write_into = result_tensor, punctuation_indicies = punctuation_indicies))
    model.run_with_hooks(clean_tokens)
    model.reset_hooks()
    return result_tensor
# %%
result_tensor = get_head_percent_attend_to_punctuation()
# %%
imshow(result_tensor, title = f"percent to which heads in {model_name} attend to punctuation")

# %%

@jaxtyped
@typechecker
def see_which_indicies_head_attended_to_punctuation(
    attn_pattern: Float[Tensor, "batch n_heads seqQ seqK"],
    hook: HookPoint,
    head_index: int,
    layer_index: int,
    tensor_to_write_into: Float[Tensor, "batch len"],
    punctuation_indicies: List[List[int]] = [] # Should be a list that holds batch amount of lists of indicies for punctuation,
):
    """
    fills out a batch x seqlen tensor for what indicies the head attended to punctuation
    """
    head_attn = attn_pattern[:, head_index, :, :]
    max_token_attended_to: Float[Tensor, "batch seqQ"] = torch.argmax(head_attn, dim=-1)
    

    for batch, p_indicies in enumerate(punctuation_indicies):
        for pos in range(PROMPT_LEN):
            if max_token_attended_to[batch, pos] in p_indicies:
                tensor_to_write_into[batch, pos] = 1


@jaxtyped
@typechecker
def generate_punc_vectors(layer, head, model = model, device = device):
    """
    gets output vectors for a head and sees creates labels for if the model attended to punctuation or not
    """
    # Pick random indices for owt_dataset tokens
    
    
    all_resids = []
    all_labels = []

    def store_item_hook(
        resid_in_model,
        hook,
        where_to_store: list
    ):
        where_to_store.append(resid_in_model.clone())


    model.reset_hooks()
    storage = []
    model.add_hook(utils.get_act_name("result", layer), partial(store_item_hook, where_to_store = storage))
    
    labels = torch.zeros((BATCH_SIZE, PROMPT_LEN)).to(device)
    model.add_hook(utils.get_act_name("pattern", layer),
                            partial(see_which_indicies_head_attended_to_punctuation, head_index = head, layer_index = layer,
                                    tensor_to_write_into = labels, punctuation_indicies = punctuation_indicies))
  
    model.run_with_hooks(clean_tokens)
    model.reset_hooks()


    head_output_vectors = storage[0][..., head, :]
    
    
       
    
    flattened_outputs = einops.rearrange(head_output_vectors, "b p h -> (b p) h")
    flattened_labels = einops.rearrange(labels, "b p -> (b p)")
    

    
    assert flattened_outputs.shape[0] == flattened_labels.shape[0]
    assert flattened_outputs.shape[1] == model.cfg.d_model
    assert len(flattened_outputs.shape) == 2
    assert len(flattened_labels.shape) == 1

    return flattened_outputs, flattened_labels


flattened_outputs, flattened_labels = generate_punc_vectors(1,1)
        

# %%

def preprocess_data(flattened_outputs, flattened_labels):
    # Convert Tensors to numpy arrays if necessary
    if isinstance(flattened_outputs, torch.Tensor):
        flattened_outputs = flattened_outputs.cpu().numpy()
    if isinstance(flattened_labels, torch.Tensor):
        flattened_labels = flattened_labels.cpu().numpy()

    # Identify the minority class and count instances
    unique, counts = np.unique(flattened_labels, return_counts=True)
    min_class = unique[np.argmin(counts)]
    min_class_count = counts[np.argmin(counts)]

    # Filter indices for both classes
    indices_0 = np.where(flattened_labels == 0)[0]
    indices_1 = np.where(flattened_labels == 1)[0]

    # If either are empty, return None
    if len(indices_0) == 0 or len(indices_1) == 0:
        return torch.zeros((1)), torch.zeros((1)), torch.zeros((1)), torch.zeros((1))

    # Randomly select instances from the majority class to match the minority class count
    if min_class == 0:
        indices_1 = np.random.choice(indices_1, min_class_count, replace=False)
    else:
        indices_0 = np.random.choice(indices_0, min_class_count, replace=False)

    # Combine and shuffle the balanced indices
    balanced_indices = np.concatenate((indices_0, indices_1))
    np.random.shuffle(balanced_indices)

    # Extract balanced outputs and labels
    balanced_outputs = flattened_outputs[balanced_indices]
    balanced_labels = flattened_labels[balanced_indices]

    # Split into training and testing sets
    split_idx = len(balanced_indices) * 2 // 3
    train_outputs = balanced_outputs[:split_idx]
    test_outputs = balanced_outputs[split_idx:]
    train_labels = balanced_labels[:split_idx]
    test_labels = balanced_labels[split_idx:]

    return train_outputs, test_outputs, train_labels, test_labels


def train_classifier_on_ablation_sklearn(model, flattened_outputs, flattened_labels):
    
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
detection_ability = torch.zeros((model.cfg.n_layers, model.cfg.n_heads))

# %%
for layer in range(0, model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        print(f"layer {layer} head {head}")
        flattened_outputs, flattened_labels = generate_punc_vectors(layer, head)
        _, accuracy = train_classifier_on_ablation_sklearn(model, flattened_outputs, flattened_labels)
        detection_ability[layer, head] = accuracy
# %%
fig = imshow(detection_ability, title = "classification accuracy of punctuation in gpt2 small", return_fig=True)
# save to html
fig.write_html(f"clustering_results/{safe_model_name}-punctuation-detection.html")


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
