# a file to automate calls to overview_self_repair

import subprocess

model_names = [
    # "gpt2-small",
    # "gpt2-medium",
    # "gpt2-large",
    # "opt-125m",
    # "opt-410m",
    # "gpt-neo-125M",
    "pythia-160m",
    "stanford-gpt2-small-a",
    "stanford-gpt2-medium-a",
    "pythia-410m",
    "opt-1.3b",
    "pythia-1b-deduped",
    "tiny-stories-instruct-33M",
]

for model_name in model_names:
    print("Running with model " + model_name)
    subprocess.run(["python", "GOOD_self_repressing_experiment.py", "--model_name=" + model_name])
