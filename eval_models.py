import subprocess
import os

folder_path = '/home/erezlid/CapDec/coco_train/noise_by_sim'
dirs = [directory for directory in os.listdir(folder_path) if 'samples' in directory]
print(dirs)

for sample_dir in dirs:
    file_path = os.path.join(sample_dir,sample_dir + '.pt')
    checkpoint_path = os.path.join(folder_path, file_path)
    job_name = f"{sample_dir}"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        '--export',
        f"CHECKPOINT_PATH={checkpoint_path}",
        "evaluation.sbatch"
    ]
    subprocess.run(cmd)

