import os
import subprocess

from constant_config import RESULTS_FOLDER,CAPDEC_RESULTS_FILE

result_files = os.listdir(RESULTS_FOLDER)


for file in result_files:
    if file == CAPDEC_RESULTS_FILE:
        continue
    print(file)
    command = ['python','plots.py','--result_file', file]
    subprocess.call(command)