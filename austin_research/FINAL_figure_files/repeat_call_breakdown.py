# a file to automate calls to python files

import subprocess

model_names = [
    #"pythia-160m", # DONE with 1_000_000
    #"gpt2-small",  # DONE with 1_000_000
    #"pythia-410m", # DONE with 1_000_000
    #"gpt2-medium", # DONE with 1_000_000
    #"gpt2-large",  # DONE with 50,000
    #"llama-7b", # queued with 25_000
    #"pythia-1b", # queed with 1_000_000
    
    
    #"opt-125m",
    #"opt-410m",
    #"gpt-neo-125M",
    #stanford-gpt2-small-a",
    #"stanford-gpt2-medium-a",
    # "pythia-410m-deduped",
    # #"opt-1.3b",
    # "pythia-1b-deduped",
    # "tiny-stories-instruct-33M",
    
]


batch_sizes = [4, 3, 2, 1]


for model_name in model_names:
    print("Running with model " + model_name)

    for batch_size in batch_sizes:  # Continue trying until batch_size is positive
        try:
            # Attempt to run the script with the current batch size
            subprocess.run(["python", "GOOD_breakdown_self_repair.py", 
                            "--model_name=" + model_name, 
                            "--batch_size=" + str(batch_size),
                            "--percentile=" + str(0.02),
                            "--min_tokens=" + str(25_000),
                            ], 
                            check=True)  # check=True makes it raise an error if the command fails
            break  # If successful, proceed to next model

        except subprocess.CalledProcessError as e:
            # If an error occurs, subtract 10 from batch size and retry
            print(f"Error occurred with batch size {batch_size}: {e}, retrying with smaller batch size...")
