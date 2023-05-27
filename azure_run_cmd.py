"""
Usage:

python azure_run_cmd.py 1_err-lstm 'nohup python3 -u train.py --model lstm --n_continuity_errors 1 &'

python azure_run_cmd.py 1_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 1 &'

python azure_run_cmd.py 1_err-textcnn 'nohup python3 -u train.py --model textcnn --n_continuity_errors 1 &'

python azure_run_cmd.py 1_err-bert_kg 'nohup python3 -u train.py --model bert_kg --n_continuity_errors 1 &'

python azure_run_cmd.py 1_err-get 'nohup python3 -u train.py --model get --n_continuity_errors 1 &'
python azure_run_cmd.py 1_err-mac 'nohup python3 -u train.py --model mac --n_continuity_errors 1 &'

NOTE: get/mac run out of memory at 16gb because we load everything.
"""

import sys
from azure_process_data import create_azure_vm
import paramiko
import time

train_image = "/subscriptions/5a5fec05-4c7d-4b5f-9142-bf8a1f62966d/resourceGroups/plot_hole_detection/providers/Microsoft.Compute/galleries/stanfordcorenlp/images/train/versions/3.0.0"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 azure_rum_cmd.py vm_name [command to run on vm]")
        sys.exit()
    name = sys.argv[1]
    cmd = " ".join(sys.argv[2:])
    print(f"Executing: {cmd}")

    # Create new vm with given name
    # Standard_D4s_v3 for all models except for GET and MAC which require E4-2ads_v5 # NOTE try Standard_E2ads_v5 for all models?
    size = (
        "Standard_E4-2ads_v5" if "mac" in cmd or "get" in cmd else "Standard_E2ads_v5"
    )
    host, username, password = create_azure_vm(
        name=name, image=train_image, size=size, use_ssh=True
    )
    port = 22

    start_time = time.time()
    # Connect to vm and start the command
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh.connect(host, port, username, password)
    ssh.connect(host, port, username, key_filename="kg_microservice/azureuser.pem")
    channel = ssh.invoke_shell()
    channel.send("cd ~/plot_hole_detection_next\n")
    channel.send(cmd + "\n")
    channel.send("\n")
    print("Running!")
    # Print output from the remote shell
    while not channel.exit_status_ready():
        if channel.recv_ready():
            print(channel.recv(1024).decode("utf-8"))
        if time.time() - start_time > 30:
            # break out if it takes too long since this thing never breaks out for some reason
            break

    print(f"Connect with: ssh -i azureuser.pem {username}@{host}")
