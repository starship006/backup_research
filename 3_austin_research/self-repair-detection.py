# %%
from imports import *
import argparse
from helpers import return_partial_functions, return_item, topk_of_Nd_tensor, residual_stack_to_direct_effect, show_input, collect_direct_effect, dir_effects_from_sample_ablating, get_correct_logit_score
# activate gradient
torch.set_grad_enabled(True)
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
owt_dataset = utils.get_dataset("owt")
owt_dataset_name = "owt"

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
def train_classifier_on_ablation(model, classifier, num_epochs=10, learning_rate=0.001, BATCH_SIZE=5, PROMPT_LEN=40):
      
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    num_heads = model.cfg.n_heads

    accuracies = []
    rolling_losses = []


    pbar = tqdm(range(num_epochs), desc="Training", dynamic_ncols=True)
    for epoch in pbar:
        all_owt_tokens = model.to_tokens(owt_dataset[(epoch * BATCH_SIZE * num_heads * 2):((epoch + 1) * BATCH_SIZE * num_heads * 2)]["text"]).to(device)

        all_resids = []
        all_labels = []

        for head_to_ablate in range(num_heads):
            start_idx = head_to_ablate * BATCH_SIZE
            end_idx = (head_to_ablate + 1) * BATCH_SIZE

            owt_tokens = all_owt_tokens[start_idx:end_idx, :PROMPT_LEN]
            corrupted_owt_tokens = all_owt_tokens[BATCH_SIZE*num_heads + start_idx:BATCH_SIZE*num_heads + end_idx, :PROMPT_LEN]
            #print(owt_tokens.shape, corrupted_owt_tokens.shape)
            # Use one-hot encoding for labels
            labels = F.one_hot(torch.tensor([head_to_ablate]), num_classes=num_heads).to(torch.long).to(device)

            _, _, new_cache = dir_effects_from_sample_ablating(model, owt_tokens, corrupted_owt_tokens, attention_heads=[(9, head_to_ablate)], return_cache=True)
            resid_pre = new_cache[utils.get_act_name("resid_pre", 10)]
            
            all_resids.append(resid_pre)
            all_labels.append(einops.repeat(labels, "1 h -> b p h", b=BATCH_SIZE, p=PROMPT_LEN))

        # Concatenate and flatten all the resids and labels
        concatenated_resids = torch.cat(all_resids, dim=0)
        concatenated_labels = torch.cat(all_labels, dim=0)
        flattened_resids = einops.rearrange(concatenated_resids, "b p h -> (b p) h").float()
        flattened_labels = einops.rearrange(concatenated_labels, "b p h -> (b p) h").float()

        optimizer.zero_grad()
        outputs = classifier(flattened_resids).to(device)
        loss = criterion(outputs, flattened_labels)
        loss.backward()
        optimizer.step()

        # get some stats
        predicted = torch.argmax(outputs, dim=-1)
        correct_preds = (predicted == torch.argmax(flattened_labels, dim=-1)).sum().item()
        accuracy = correct_preds / (BATCH_SIZE * PROMPT_LEN * num_heads)
        accuracies.append(accuracy)
        rolling_losses.append(loss.item())

        #if epoch % 10 == 9:
        #    print(f"Epoch {epoch+1}, Rolling Accuracy: {sum(accuracies[-10:]) / 10}, Rolling Loss: {sum(rolling_losses[-10:]) / 10}")
        pbar.set_postfix({"Current Accuracy": f"{accuracy:.5f}", "Current Loss": f"{loss.item():.5f}"})
    return accuracies

        
# %%
input_dim = model.cfg.d_model
num_heads = model.cfg.n_heads  # * model.cfg.n_layers  # Total number of heads across all layers
classifier = MLPClassifier(input_dim, num_heads, [600, 20]).to(device)  


# %%
accuracies = train_classifier_on_ablation(model, classifier, num_epochs=100, learning_rate=0.001, BATCH_SIZE = 10, PROMPT_LEN=100)


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
for head in range(model.cfg.n_heads):
    evaluate_classifier_on_specific_head(model, classifier, head, num_evals = 20, BATCH_SIZE = 40, PROMPT_LEN = 2)
# %%
