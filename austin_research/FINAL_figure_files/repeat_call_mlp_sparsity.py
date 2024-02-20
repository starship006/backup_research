# a file to automate calls to python files

import subprocess



model_names_layer_head_tokens = [
    ("pythia-160m", 9, 3, 1_000_000),
    ("gpt2-small", 10, 2, 1_000_000),
    ("pythia-410m", 20, 6, 1_000_000),
    ("llama-7b", 30, 8, 1_000_000),
    #"gpt2-medium",
    #"gpt2-large",
    #"opt-125m",
    #"opt-410m",
    #"gpt-neo-125M",
    #stanford-gpt2-small-a",
    #"stanford-gpt2-medium-a",
    #"pythia-410m",
    # "pythia-410m-deduped",
    # #"opt-1.3b",
    # "pythia-1b",
    # "pythia-1b-deduped",
    # "tiny-stories-instruct-33M",
]


batch_sizes = [25, 15, 5, 4, 3, 2, 1] #[5, 4, 3, 2, 1]## #


for model_tuple in model_names_layer_head_tokens:
    model_name, layer, head, num_tokens = model_tuple
    
    print("Running with model " + model_name)
    
    for batch_size in batch_sizes:  # Continue trying until batch_size is positive
        try:
            # Attempt to run the script with the current batch size
            subprocess.run(["python", "GOOD_MLP_erasure_breakdown.py", 
                            "--model_name=" + model_name, 
                            "--batch_size=" + str(batch_size),
                            "--head=" + str(head),
                            "--ablate_layer=" + str(layer),
                            "--min_tokens=" + str(num_tokens),
                            ], 
                            check=True)  # check=True makes it raise an error if the command fails
            break  # If successful, proceed to next model
        except subprocess.CalledProcessError as e:
            # If an error occurs, subtract 10 from batch size and retry
            print(f"Error occurred with batch size {batch_size}: {e}, retrying with smaller batch size...")
            