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
    
def setup_kaggle():
    cmd1 = 'pip install --upgrade --force-reinstall --no-deps kaggle'
    cmd2 = 'mkdir /root/.kaggle'

    execute_shell(cmd1)
    execute_shell(cmd2)

    with open("/root/.kaggle/kaggle.json", "w+") as f:
        f.write('{"username":"<kaggle-username>","key":"<kaggle-key>"}') # Put your kaggle username & key here

    cmd3 = 'chmod 600 /root/.kaggle/kaggle.json'
    execute_shell(cmd3)