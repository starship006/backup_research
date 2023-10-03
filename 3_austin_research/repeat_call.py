# a file to automate calls to overview_self_repair

import subprocess

model_names = [
    "gpt2-medium",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "roneneldan/TinyStories-33M",
    #"NeelNanda/SoLU_12L1536W_C4_Code",
]

for model_name in model_names:
    subprocess.run(["python", "overview_self_repair.py", "--model_name=" + model_name])
