# a file to automate calls to python files

import subprocess

model_names = [
    "pythia-160m",
    "gpt2-small",
    #"gpt2-medium",
    #"gpt2-large",
    #"opt-125m",
    #"opt-410m",
    #"gpt-neo-125M",
    #stanford-gpt2-small-a",
    #"stanford-gpt2-medium-a",
    "pythia-410m",
    # "pythia-410m-deduped",
    # #"opt-1.3b",
    # "pythia-1b",
    # "pythia-1b-deduped",
    # "tiny-stories-instruct-33M",
]


initial_batch_size = 30  # Starting point for batch size

for model_name in model_names:
    print("Running with model " + model_name)
    batch_size = initial_batch_size  # Initialize batch size for each model

    while batch_size > 0:  # Continue trying until batch_size is positive
        try:
            # Attempt to run the script with the current batch size
            subprocess.run(["python", "GOOD_self_repair_graph_generator.py", 
                            "--model_name=" + model_name, 
                            "--batch_size=" + str(batch_size),
                            "--ablation_type=" + "mean"], 
                            check=True)  # check=True makes it raise an error if the command fails
            break  # If successful, proceed to next model

        except subprocess.CalledProcessError as e:
            # If an error occurs, subtract 10 from batch size and retry
            print(f"Error occurred with batch size {batch_size}: {e}, retrying with smaller batch size...")
            batch_size -= 10

    if batch_size <= 0:
        print(f"Failed to run model {model_name} with any batch size, moving to next model.")