# a file to automate calls to overview_self_repair

import subprocess

model_names = [
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "NeelNanda/SoLU_8L_v21_old",
    "NeelNanda/SoLU_10L_v22_old",
    "NeelNanda/SoLU_12L_v23_old"
]

for model_name in model_names:
    subprocess.run(["python", "overview_self_repair.py", "--model_name=" + model_name])
