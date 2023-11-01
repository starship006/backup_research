# a file to automate calls to overview_self_repair

import subprocess

model_names = [
    "gpt2-small",
    "gpt2-medium",
    "opt-125m",
    "gpt-neo-125M",
    "pythia-160m",
    "stanford-gpt2-small-a",
    "stanford-gpt2-medium-a",
    "pythia-410m",
    "solu-12l-pile",
    "bert-base-cased",
    

    #"NeelNanda/SoLU_12L1536W_C4_Code",
]

for model_name in model_names:
    subprocess.run(["python", "self-repair-detection.py", "--model_name=" + model_name])
