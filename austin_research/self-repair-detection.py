# %%
from imports import *
import argparse
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
from sklearn.linear_model import LogisticRegression
from beartype import beartype as typechecker


# %%
# activate gradient
torch.set_grad_enabled(True)
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
owt_dataset = utils.get_dataset("owt")
owt_dataset_name = "owt"

# %%
pile_dataset = utils.get_dataset("pile")
pile_dataset_name = "pile"



# %%

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_heads)
        
    def forward(self, x):
        x = self.fc(x)
        return x
    
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dims):
        super(MLPClassifier, self).__init__()
        
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, num_heads))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    

# %%
import random

def generate_data(model, owt_dataset, BATCH_SIZE, num_heads, PROMPT_LEN, device, layer_to_ablate=9):
    """
    generates residuals of flattened residual streams when specific heads are ablated
    picks random prompts for each epoch
    each ablated head has PROMPT_LEN * BATCH_SIZE residual stream vectors generated for it
    """
    # Pick random indices for owt_dataset tokens
    random_indices = random.sample(range(len(owt_dataset)), BATCH_SIZE * num_heads * 2)
    all_owt_tokens = model.to_tokens([owt_dataset[i]["text"] for i in random_indices]).to(device)
    
    all_resids = []
    all_labels = []
    
    for head_to_ablate in range(num_heads):
        # generate the residual stream vectors
        start_idx = head_to_ablate * BATCH_SIZE
        end_idx = (head_to_ablate + 1) * BATCH_SIZE
        owt_tokens = all_owt_tokens[start_idx:end_idx, :PROMPT_LEN]
        corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE * num_heads + start_idx:BATCH_SIZE * num_heads + end_idx, :PROMPT_LEN]
        
        assert owt_tokens.shape == corrupted_owt_tokens.shape
        
        _, _, new_cache = dir_effects_from_sample_ablating(model, owt_tokens, corrupted_owt_tokens, attention_heads=[(layer_to_ablate, head_to_ablate)], return_cache=True)
        resid_pre = new_cache[utils.get_act_name("resid_pre", 10)]
        all_resids.append(resid_pre)
        
        # generate labels
        labels = F.one_hot(torch.tensor([head_to_ablate]), num_classes=num_heads).to(device).float()
        all_labels.append(einops.repeat(labels, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))
        
    concatenated_resids = torch.cat(all_resids, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)
    
    flattened_resids = einops.rearrange(concatenated_resids, "b p h -> (b p) h")
    flattened_labels = einops.rearrange(concatenated_labels, "b p h -> (b p) h")
    
    assert flattened_resids.shape[0] == flattened_labels.shape[0]
    assert len(flattened_resids.shape) == 2
    
    return flattened_resids, flattened_labels


@jaxtyped
@typechecker
def generate_isolated_vectors(model, BATCH_SIZE: int, num_heads: int, PROMPT_LEN: int, 
                              device: str, zero_ablate: bool, layer_to_ablate: int =9, 
                              act_to_read:str = utils.get_act_name("resid_mid", 9), dataset: datasets.arrow_dataset.Dataset = owt_dataset):
    """
    vector = original vector -  ablated residual vector
    picks random prompts for each epoch
    each ablated head has PROMPT_LEN * BATCH_SIZE residual stream vectors generated for it

    zero_ablate: whether or not to zero or sample ablate the head
    """
    # Pick random indices for owt_dataset tokens
    random_indices = random.sample(range(len(dataset)), BATCH_SIZE * num_heads * 2)
    all_owt_tokens = model.to_tokens([dataset[i]["text"] for i in random_indices]).to(device)
    
    all_resids = []
    all_labels = []
    
    for head_to_ablate in range(num_heads):
        # generate the residual stream vectors
        start_idx = head_to_ablate * BATCH_SIZE
        end_idx = (head_to_ablate + 1) * BATCH_SIZE
        owt_tokens = all_owt_tokens[start_idx:end_idx, :PROMPT_LEN]
        corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE * num_heads + start_idx:BATCH_SIZE * num_heads + end_idx, :PROMPT_LEN]
        
        assert owt_tokens.shape == corrupted_owt_tokens.shape
        
        _, _, new_cache = dir_effects_from_sample_ablating(model, owt_tokens, corrupted_owt_tokens, attention_heads=[(layer_to_ablate, head_to_ablate)], return_cache=True)
        resid_pre = new_cache[act_to_read]

        # get the original vectors
        _, clean_cache = model.run_with_cache(owt_tokens)
        clean_resid_pre = clean_cache[act_to_read]

        # subtract the residual vectors and add to list
        all_resids.append(clean_resid_pre - resid_pre)
        
        # generate labels
        labels = F.one_hot(torch.tensor([head_to_ablate]), num_classes=num_heads).to(device).float()
        all_labels.append(einops.repeat(labels, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))
        
    concatenated_resids = torch.cat(all_resids, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)
    
    flattened_resids = einops.rearrange(concatenated_resids, "b p h -> (b p) h")
    flattened_labels = einops.rearrange(concatenated_labels, "b p h -> (b p) h")
    
    assert flattened_resids.shape[0] == flattened_labels.shape[0]
    assert len(flattened_resids.shape) == 2
    
    return flattened_resids, flattened_labels

# %%
def generate_test_data(model, owt_dataset, epoch, BATCH_SIZE, num_heads, PROMPT_LEN, device, layer_to_ablate = 9):
    """
    generates 'resudial streams' of just one hot vectors; data should be easily predictable
    """
    all_owt_tokens = model.to_tokens(owt_dataset[(epoch * BATCH_SIZE * num_heads * 2):((epoch + 1) * BATCH_SIZE * num_heads * 2)]["text"]).to(device)
    all_resids = []
    all_labels = []
    for head_to_ablate in range(num_heads):
        # generate new residual stream vectors
        resids = F.one_hot(torch.tensor([head_to_ablate]), num_classes=model.cfg.d_model).to(torch.long).to(device)
        all_resids.append(einops.repeat(resids, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))
        # generate labels
        labels = F.one_hot(torch.tensor([head_to_ablate]), num_classes=num_heads).to(torch.long).to(device)
        all_labels.append(einops.repeat(labels, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))

    concatenated_resids = torch.cat(all_resids, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)
    flattened_resids = einops.rearrange(concatenated_resids, "b p h -> (b p) h").float()
    flattened_labels = einops.rearrange(concatenated_labels, "b p h -> (b p) h").float()
    
    assert flattened_resids.shape[0] == flattened_labels.shape[0]
    assert len(flattened_resids.shape) == 2
    #print(flattened_resids[0:10], flattened_resids[0 + BATCH_SIZE * PROMPT_LEN:10+ BATCH_SIZE * PROMPT_LEN])
    return flattened_resids, flattened_labels
# %%
def train_classifier_on_ablation_sklearn(model, dataset_func, BATCH_SIZE=5, PROMPT_LEN=40, patience=5, min_train=40, test_size=100):
    classifier = LogisticRegression()  # or any other scikit-learn classifier
    best_accuracy = 0
    epochs_no_improve = 0
    epoch = 0
    
    while True:
        flattened_resids, flattened_labels = dataset_func(model, 100, model.cfg.n_heads, PROMPT_LEN, device, True, 
                                                                       layer_to_ablate=9, act_to_read = utils.get_act_name("resid_post", 9), dataset = owt_dataset)
        
        # Reshape labels for scikit-learn
        labels = np.argmax(flattened_labels.cpu().numpy(), axis=1)
        print(labels)
        
        # Train the classifier
        classifier.fit(flattened_resids.cpu().numpy(), labels)
        
        # Predict and calculate accuracy on training set
        predicted = classifier.predict(flattened_resids.cpu().numpy())
        accuracy = accuracy_score(labels, predicted)

        # Get Test Accuracy
        test_resids, test_labels = dataset_func(model,30, model.cfg.n_heads, PROMPT_LEN, device, True, 
                                                     layer_to_ablate=9, act_to_read = utils.get_act_name("resid_post", 9), dataset = owt_dataset)
        test_labels = np.argmax(test_labels.cpu().numpy(), axis=1)
        test_predicted = classifier.predict(test_resids.cpu().numpy())
        test_accuracy = accuracy_score(test_labels, test_predicted)
        
        print(f"Epoch {epoch}, Training Accuracy: {accuracy:.5f}", f"Test Accuracy: {test_accuracy:.5f}")
        print("Classification Report:\n", classification_report(test_labels, test_predicted))
        
        # Early stopping logic
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience and epoch > min_train:
            print(f"Early stopping triggered: no improvement in accuracy for {patience} epochs")
            break

        epoch += 1
    

    return classifier  # return the trained model






    

def train_classifier_on_ablation(model, classifier, dataset_func, learning_rate=0.01, BATCH_SIZE=5, PROMPT_LEN=40, patience=5, min_train = 40):
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_heads = model.cfg.n_heads
    accuracies = []
    rolling_losses = []
    pbar = tqdm(desc="Training", dynamic_ncols=True)
    
    best_loss = float('inf')
    epochs_no_improve = 0
    
    epoch = 0
    while True:  # replace fixed num_epochs with an open-ended loop
        flattened_resids, flattened_labels = dataset_func(model, owt_dataset, BATCH_SIZE, num_heads, PROMPT_LEN, device)
        optimizer.zero_grad()
        outputs = classifier(flattened_resids).to(device)
        loss = criterion(outputs, flattened_labels)
        loss.backward()
        optimizer.step()

        predicted = torch.argmax(outputs, dim=-1)
        correct_preds = (predicted == torch.argmax(flattened_labels, dim=-1)).sum().item()
        accuracy = correct_preds / (BATCH_SIZE * PROMPT_LEN * num_heads)
        accuracies.append(accuracy)
        rolling_losses.append(loss.item())
        
        pbar.set_postfix({"Current Accuracy": f"{accuracy:.5f}", "Current Loss": f"{loss.item():.5f}"})
        pbar.update(1)
        
        # Check if the loss has stopped decreasing
        if loss.item() < best_loss:
            best_loss = loss.item()
            epochs_no_improve = 0  # reset
        else:
            epochs_no_improve += 1
        
        # If loss hasn't improved for "patience" epochs, stop training
        if epochs_no_improve >= patience and epoch > min_train:
            print(f"Early stopping triggered: no improvement in loss for {patience} epochs")
            break

        epoch += 1  # increment epoch counter for open-ended loop

    pbar.close()
    return accuracies

def save_classifier_weights(classifier, model_name, detail, folder = "default"):
    if folder:
        save_path = f'signature_weights/{folder}/{model_name}_{safe_model_name}_{detail}_classifier.pt'
    else:
        save_path = f'signature_weights/{model_name}_{safe_model_name}_{detail}_classifier.pt'
   
    torch.save(classifier.state_dict(), save_path)

def load_classifier_weights(model_name, input_dim, num_heads, hidden_dims, device, detail, folder = "default"):
    if folder:
        load_path = f'signature_weights/{folder}/{model_name}_{safe_model_name}_{detail}_classifier.pt'
    else:
        load_path = f'signature_weights/{model_name}_{safe_model_name}_{detail}_classifier.pt'
    classifier = None

    if model_name == 'MLPClassifier':
        classifier = MLPClassifier(input_dim, num_heads, hidden_dims).to(device)
    elif model_name == 'LinearClassifier':
        classifier = LinearClassifier(input_dim, num_heads).to(device)
    else:
        raise ValueError(f"Unsupported model_name {model_name}")

    try:
        classifier.load_state_dict(torch.load(load_path))
    except FileNotFoundError:
        print(f"No saved weights found for {model_name}. Initializing with random weights.")

    return classifier

def train_tons_of_classifiers(rank, world_size):
    # Initialize process group
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #dpp_model = DPP(model, device_ids = [rank])
    
    ablate_by_read_classifier_accuracy = torch.zeros((model.cfg.n_layers, model.cfg.n_layers))
    for ablate_layer in range(model.cfg.n_layers):
        for read_layer in range(ablate_layer, model.cfg.n_layers):
            print("Ablating layer ", ablate_layer, " and reading layer ", read_layer)
            detail = f"ablate_{ablate_layer}_read_{read_layer}"
            input_dim = model.cfg.d_model
            num_heads = model.cfg.n_heads
            hidden_dims = [400, 100]
            device = 'cuda'  # Replace this with your actual device

            classifier = load_classifier_weights("MLPClassifier", input_dim, num_heads, hidden_dims, device, detail)

            # put the classifier on the GPU
            

            dataset_func = partial(generate_isolated_vectors, layer_to_ablate = ablate_layer, act_to_read = utils.get_act_name("resid_post", read_layer) ,zero_ablate=False)


            accuracies = train_classifier_on_ablation(model, classifier, dataset_func, learning_rate=0.01, BATCH_SIZE = 10, PROMPT_LEN=50)
            
            # get average of last 5 accuracies
            ablate_by_read_classifier_accuracy[ablate_layer, read_layer] = sum(accuracies[-5:]) / 5
            torch.cuda.empty_cache()
            save_classifier_weights(classifier, "MLPClassifier", detail)


    print(ablate_by_read_classifier_accuracy)
    fig = imshow(ablate_by_read_classifier_accuracy, title = f"Accuracy of classifiers predicting ablated heads in {model_name}",
       text_auto = True, width = 800, height = 800, return_fig=True)

    # save figure to html
    fig.write_html(f"threshold_figures/{safe_model_name}_ablate_by_read_classifier_accuracy.html")
# %%


train_classifier_on_ablation_sklearn(model, generate_isolated_vectors)

# run single GPU of train_tons_of_classifiers

# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "29500"

# if __name__ == '__main__':
#     world_size = 4
#     torch.multiprocessing.spawn(train_tons_of_classifiers, args=(world_size,), nprocs=world_size, join=True)
#train_tons_of_classifiers(0, 1)


# %% 
def generate_mlp_vectors(model, owt_dataset, BATCH_SIZE, num_heads, PROMPT_LEN, device, zero_ablate, layer_to_ablate=9, act_to_read = utils.get_act_name("post", 9), dataset = owt_dataset, diff_vector = False):
    """
    ablates MLP neurons and gets activations that result from the activations
    picks random prompts for each epoch
    each ablated head has PROMPT_LEN * BATCH_SIZE residual stream vectors generated for it

    diff_vector; whether or not to generate vectors which are differences of activations, as opposed to just the activations
    """
    # Pick random indices for owt_dataset tokens
    random_indices = random.sample(range(len(dataset)), BATCH_SIZE * num_heads * 2)
    all_owt_tokens = model.to_tokens([dataset[i]["text"] for i in random_indices]).to(device)
    
    all_resids = []
    all_labels = []
    
    for head_to_ablate in range(num_heads):
        # generate the residual stream vectors
        start_idx = head_to_ablate * BATCH_SIZE
        end_idx = (head_to_ablate + 1) * BATCH_SIZE
        owt_tokens = all_owt_tokens[start_idx:end_idx, :PROMPT_LEN]
        corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE * num_heads + start_idx:BATCH_SIZE * num_heads + end_idx, :PROMPT_LEN]
        
        assert owt_tokens.shape == corrupted_owt_tokens.shape
        
        _, _, new_cache = dir_effects_from_sample_ablating(model, owt_tokens, corrupted_owt_tokens, attention_heads=[(layer_to_ablate, head_to_ablate)], return_cache=True)
        new_activations = new_cache[act_to_read]

        # get the original vectors
        _, clean_cache = model.run_with_cache(owt_tokens)
        clean_activations = clean_cache[act_to_read]

        # subtract the residual vectors and add to list
        if diff_vector:
            # _, _, new_cache = dir_effects_from_sample_ablating(model, owt_tokens, corrupted_owt_tokens, attention_heads=[(layer_to_ablate, head_to_ablate)], return_cache=True)
            # activations = new_cache[act_to_read]
            all_resids.append(clean_activations - new_activations)
        else:
            all_resids.append(new_activations)
        
        # generate labels
        labels = F.one_hot(torch.tensor([head_to_ablate]), num_classes=num_heads).to(device).float()
        all_labels.append(einops.repeat(labels, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))
        
    concatenated_resids = torch.cat(all_resids, dim=0)
    concatenated_labels = torch.cat(all_labels, dim=0)
    
    flattened_resids = einops.rearrange(concatenated_resids, "b p h -> (b p) h")
    flattened_labels = einops.rearrange(concatenated_labels, "b p h -> (b p) h")
    
    assert flattened_resids.shape[0] == flattened_labels.shape[0]
    assert len(flattened_resids.shape) == 2
    
    return flattened_resids, flattened_labels


# %%
def train_tons_of_mlp_classifiers(rank, world_size, type_a = "pure"):
    # Initialize process group
    #dist.init_process_group("gloo", rank=rank, world_size=world_size)
    #dpp_model = DPP(model, device_ids = [rank])
    assert (type_a == "pure" or type_a == "diff")
    if type_a == "pure":
        diff = False
    else:
        diff = True


    ablate_by_read_classifier_accuracy = torch.zeros((model.cfg.n_layers, model.cfg.n_layers))
    for ablate_layer in range(8, model.cfg.n_layers):
        for read_layer in range(ablate_layer, ablate_layer + 1):#model.cfg.n_layers):
            print("Ablating MLP layer ", ablate_layer, " and reading MLP layer ", read_layer)
            detail = f"MLP_ablate_{ablate_layer}_read_{read_layer}"
            input_dim = model.cfg.d_mlp
            num_heads = model.cfg.n_heads
            hidden_dims = [1500, 500, 100]
            device = 'cuda'  # Replace this with your actual device

            classifier = load_classifier_weights("MLPClassifier", input_dim, num_heads, hidden_dims, device, detail, folder = f"mlp_{type_a}_activs")

            # put the classifier on the GPU
            
            dataset_func = partial(generate_mlp_vectors, layer_to_ablate = ablate_layer, act_to_read = utils.get_act_name("post", read_layer) ,zero_ablate=False, diff_vector=diff)
            accuracies = train_classifier_on_ablation(model, classifier, dataset_func, learning_rate=0.01, BATCH_SIZE = 10, PROMPT_LEN=50, min_train = 20)
            # get average of last 5 accuracies
            ablate_by_read_classifier_accuracy[ablate_layer, read_layer] = sum(accuracies[-5:]) / 5
            torch.cuda.empty_cache()
            save_classifier_weights(classifier, "MLPClassifier", detail, folder = f"mlp_{type_a}_activs")


    print(ablate_by_read_classifier_accuracy)
    fig = imshow(ablate_by_read_classifier_accuracy, title = f"Accuracy of classifiers predicting ablated heads in {model_name}",
       text_auto = True, width = 800, height = 800, return_fig=True)
    fig.show()

    # save figure to html
    fig.write_html(f"threshold_figures/{safe_model_name}_ablate_by_read_MLP_classifier_accuracy.html")


#%%
train_tons_of_mlp_classifiers(0, 1, type_a = "pure")

# %%
# Create classifier
# %%
save_classifier_weights(classifier, classifier_type)
# %%
def evaluate_classifier_on_specific_head(model, classifier, head_to_ablate, num_evals=100, BATCH_SIZE=5, PROMPT_LEN=40):
    num_heads = model.cfg.n_heads
    total_correct = 0
    total_count = 0

    predictions = torch.zeros(num_heads)

    for i in range(num_evals):
        all_owt_tokens = model.to_tokens(owt_dataset[(i * BATCH_SIZE * 2):((i + 1) * BATCH_SIZE * 2)]["text"]).to(device)
        owt_tokens = all_owt_tokens[0:BATCH_SIZE][:, :PROMPT_LEN]
        corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE:BATCH_SIZE * 2][:, :PROMPT_LEN]
        
        _, _, new_cache = dir_effects_from_sample_ablating(model, owt_tokens, corrupted_owt_tokens, attention_heads=[(9, head_to_ablate)], return_cache=True)

        with torch.no_grad():
            outputs = classifier(new_cache[utils.get_act_name("resid_pre", 10)]).to(device)
            predicted = torch.argmax(outputs, dim=-1)
            
            # Tally up the predictions
            for head in predicted.view(-1).cpu().numpy():
                predictions[head] += 1

            total_correct += (predicted == head_to_ablate).sum().item()
            total_count += BATCH_SIZE * PROMPT_LEN

    accuracy = total_correct / total_count
    print(f"Accuracy for ablated head {head_to_ablate}: {accuracy * 100:.2f}%")
    print(f"Total Predictions across all heads: {predictions}")
    
    return accuracy

# %%

def test_classifier_on_ablations(model, classifier, dataset_func, num_epochs=10, learning_rate=0.01, BATCH_SIZE=5, PROMPT_LEN=40):
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_heads = model.cfg.n_heads
    overall_accuracies = []
    pbar = tqdm(range(num_epochs), desc="Training", dynamic_ncols=True)
    with torch.no_grad():
        for epoch in pbar:
            flattened_resids, flattened_labels = dataset_func(model, owt_dataset, BATCH_SIZE, num_heads, PROMPT_LEN, device)
            outputs = classifier(flattened_resids).to(device)
            predicted = torch.argmax(outputs, dim=-1)
            true_labels = torch.argmax(flattened_labels, dim=-1)

            corrction_predictions = (predicted == true_labels)
            correct_pred_count = corrction_predictions.sum().item()
            accuracy = correct_pred_count / (BATCH_SIZE * PROMPT_LEN * num_heads)
            overall_accuracies.append(accuracy)


            accuracy = torch.zeros(num_heads)
            precision = torch.zeros(num_heads)
            recall = torch.zeros(num_heads)


            # look at accuracy, recall, and precision for each head
            for head in range(model.cfg.n_heads):
                head_idx_start = head * BATCH_SIZE * PROMPT_LEN
                head_idx_end = (head + 1) * BATCH_SIZE * PROMPT_LEN
               
                head_true_labels = true_labels[head_idx_start:head_idx_end]
                head_predicted = predicted[head_idx_start:head_idx_end]
                

                total_correct_for_head = corrction_predictions[head * BATCH_SIZE * PROMPT_LEN:(head + 1) * BATCH_SIZE * PROMPT_LEN].sum().item()
                accuracy_head = total_correct_for_head/ (BATCH_SIZE * PROMPT_LEN)
                accuracy[head] = accuracy_head
                #print(f"Accuracy for {head}: {accuracy_head:.5f}")               
                #print(TP_head, FP_head, FN_head, TN_head)
                #print(head_true_labels[0:10], head_predicted[0:10])


            # calculate precision, recall, and f1 score for each head
            for head in range(model.cfg.n_heads):
                TP_head = ((predicted == head) & (true_labels == head)).sum().item()
                FP_head = ((predicted == head) & (true_labels != head)).sum().item()
                FN_head = ((predicted != head) & (true_labels == head)).sum().item()
                TN_head = ((predicted != head) & (true_labels != head)).sum().item()
                

                precision_head = TP_head / (TP_head + FP_head) if TP_head + FP_head > 0 else 0
                recall_head = TP_head / (TP_head + FN_head) if TP_head + FN_head > 0 else 0

                precision[head] = precision_head
                recall[head] = recall_head
                print(f"Head {head} - Accuracy {accuracy[head] * 100:.2f}% - Precision {precision_head * 100:.2f}% - Recall {recall_head * 100:.2f}%")



                
               

    return overall_accuracies


# %%
test_classifier_on_ablations(model, classifier, partial(dataset_func, dataset = pile_dataset), num_epochs = 1, BATCH_SIZE = 50, PROMPT_LEN = 50)
# %%
