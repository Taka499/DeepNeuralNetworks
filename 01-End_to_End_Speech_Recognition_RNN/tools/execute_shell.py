import os
from tqdm import tqdm
import subprocess
import shlex
import time

def execute_shell(cmd):
    print(f"Running Command: '{cmd}'")
    with tqdm(total=300, miniters=1, desc="Elapsed Time (in second)", leave=True, position=0) as t:
        process = subprocess.Popen(shlex.split(cmd), universal_newlines=True, stdout=subprocess.PIPE)
        while process.poll() is None:
            time.sleep(1)
            t.update()
        process.wait()
    print("Completed!!!\n")
    