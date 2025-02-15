import subprocess
from itertools import product
min_sim_list = [0.7,0.75,0.8,0.85,0.9,0.95]
beta_list = [0.01,0.05,0.1,0.2]
samples_list = [100,200,300,400]

combis = product(min_sim_list, beta_list, samples_list)
print(combis)
for idx,comb in enumerate(combis):
    min_sim = comb[0]
    beta = comb[1]
    sample = comb[2]
    job_name = f"N{sample}_m{min_sim}_b{beta}"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        "--export",
        f"SAMPLES={sample} BETA={beta} MINSIM={min_sim}",
        "training.sbatch"
    ]
    subprocess.run(cmd)


