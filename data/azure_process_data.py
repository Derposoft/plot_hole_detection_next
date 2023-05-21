"""
Helper script that calls data generation code on batches of 100 in azure VMs
because it takes too long to do it all locally.
"""

import argparse
import json
import math
from multiprocessing import Pool
import os
import paramiko
import secrets
import shutil
import subprocess


def create_azure_vm(resource_group, vm_name, location, admin_username, admin_password):
    # Create Azure VM
    create_command = f"az vm create --resource-group {resource_group} --name {vm_name} --location {location} --admin-username {admin_username} --admin-password {admin_password} --output json"
    create_output = subprocess.check_output(create_command, shell=True).decode("utf-8")

    # Extract public IP from the output
    create_json = json.loads(create_output)
    public_ip = create_json["publicIpAddress"]

    return public_ip


def create_azure_vm(resource_group="plot_hole_detection", key_path="../azureuser.pem"):
    # Create an azure VM with the given parameters
    random_name = secrets.token_hex(5)
    admin_username = "azureuser"
    admin_password = secrets.token_hex(10)
    cmd = [
        "az vm create",
        f"--resource-group {resource_group}",
        f"--name {random_name}",
        "--image /subscriptions/5a5fec05-4c7d-4b5f-9142-bf8a1f62966d/resourceGroups/plot_hole_detection/providers/Microsoft.Compute/galleries/stanfordcorenlp/images/stanfordcorenlp/versions/2.0.0",
        # f"--ssh-key-value {key_path}",
        # "--authentication-type ssh",
        f"--admin-username {admin_username}",
        f"--admin-password {admin_password}",
        "--public-ip-sku Basic",
        # "--size TODO", # Default size should be OK for us
    ]
    results = subprocess.check_output(" ".join(cmd), shell=True).decode("utf-8")
    results = json.loads(results)
    ip = results["publicIpAddress"]

    # Return the public IP of the azure vm
    return ip, admin_username, admin_password


def process_data_batch(docs_path, batch_id):
    # Step 1: create azure VM using our golden image
    host, username, password = create_azure_vm()
    port = 22

    # Step 2: scp docs into azure VM
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    scp = paramiko.SFTPClient.from_transport(ssh.get_transport())
    scp.put(docs_path, "data/")

    # Step 3: run command to process data in azure
    command = "python3 create_knowledge_graph.py data/ --name batch"
    stdin, stdout, stderr = ssh.exec_command(command)

    # Step 4: scp result back to local
    results_file = os.path.join(docs_path, f"knowledge_graphs_{batch_id}.pkl")
    scp.get("knowledge_graphs_batch.pkl", results_file)
    return results_file


def stitch_processed_data_batches(results):
    # Stitch together all of the processed data batches that were outputted from process_data_batch calls
    output = ""
    print(f"stitching together: {results}")
    return output


def process_all_data(data_path, batch_size=100):
    files = os.listdir(data_path)
    n_files = len(files)
    n_batches = math.ceil(n_files / batch_size)

    # Copy files in each batch to a temporary directory
    temp_dirs = []
    for batch_idx in range(n_batches):
        docs = files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        tempdir = f"temp_{batch_idx}"
        os.mkdir(tempdir)
        for doc in docs:
            shutil.copy(os.path.join(data_path, doc), tempdir)
        temp_dirs.append(tempdir)

    # Fire off subprocesses to process this batch in an azure VM. All of the work is
    # happening remotely, so it's ok to have a large number of processes in the pool.
    with Pool(len(temp_dirs)) as pool:
        all_results = pool.starmap(
            process_data_batch, zip(temp_dirs, list(range(len(temp_dirs))))
        )

    # Stitch together data batches
    result_file = stitch_processed_data_batches(all_results)
    return result_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Usage: python3 azure_process_data.py path/to/input/data"
    )
    parser.add_argument(
        "input_dir",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=10, required=False)
    args = parser.parse_args()

    # Process all data in the given input directory at the given batch size
    process_all_data(args.input_dir, batch_size=args.batch_size)
