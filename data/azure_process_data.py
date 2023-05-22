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
import pickle as pkl
import random
import scp
import secrets
import shutil
import string
import subprocess
import sys


def generate_password():
    lowercase = string.ascii_lowercase
    uppercase = string.ascii_uppercase
    digits = string.digits
    password_characters = (
        random.sample(lowercase, 1)
        + random.sample(uppercase, 1)
        + random.sample(digits, 1)
    )
    password_characters += random.sample(string.ascii_letters + string.digits, 17)
    random.shuffle(password_characters)
    password = "".join(password_characters)
    return password


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
    admin_password = generate_password()
    cmd = [
        "az vm create",
        f"--resource-group {resource_group}",
        f"--name {random_name}",
        "--image /subscriptions/5a5fec05-4c7d-4b5f-9142-bf8a1f62966d/resourceGroups/plot_hole_detection/providers/Microsoft.Compute/galleries/stanfordcorenlp/images/stanfordcorenlp/versions/3.1.0",
        # f"--ssh-key-value {key_path}", # TODO: we probably don't need keys, username password should be ok for ephemeral VMs
        # "--authentication-type ssh",
        f"--admin-username {admin_username}",
        f"--admin-password {admin_password}",
        "--public-ip-sku Basic",
        "--security-type trustedlaunch",
        "--size Standard_D2s_v3",
    ]
    results = subprocess.check_output(" ".join(cmd), shell=True).decode("utf-8")
    results = json.loads(results)
    ip = results["publicIpAddress"]
    print(
        f"Generated VM IP: {ip}, username: {admin_username}, password: {admin_password}"
    )

    # Return the public IP of the azure vm
    return ip, admin_username, admin_password


def process_data_batch(docs_path, batch_id):
    # Step 1: create azure VM using our golden image
    host, username, password = create_azure_vm()
    # host, username, password = (
    #    "20.127.22.15",
    #    "azureuser",
    #    "hT5NlpeDLSbPmnYfv4JI",
    # )  # TODO remove this debug code
    port = 22

    # Step 2: scp docs into azure VM
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    scp_client = scp.SCPClient(ssh.get_transport())
    scp_client.put(docs_path, "data/", recursive=True)

    # Step 3: run command to process data in azure
    command = "python3 create_kg.py data/ --name batch"
    stdin, stdout, stderr = ssh.exec_command(command)
    print(stdout.read().decode("utf-8"))
    print(stderr.read().decode("utf-8"))

    # Step 4: scp result back to local
    results_file = os.path.join(docs_path, f"knowledge_graphs_{batch_id}.pkl")
    scp_client.get("knowledge_graphs_batch.pkl", results_file)
    scp_client.close()
    return results_file


def stitch_processed_data_batches(results):
    # Stitch together all of the processed data batches that were outputted from process_data_batch calls
    print(f"stitching together: {results}")
    all_kgs, all_docs = [], []
    for result in results:
        with open(result, "rb") as f:
            kgs, docs = pkl.load(f)
            all_kgs += kgs
            all_docs += docs
    return all_kgs, all_docs


def process_all_data(data_path, batch_size=100, tempdirs="temp"):
    files = os.listdir(data_path)
    n_files = len(files)
    n_batches = math.ceil(n_files / batch_size)

    # Copy files in each batch to a temporary directory
    if not os.path.exists(tempdirs):
        os.mkdir(tempdirs)
    temp_dirs = []
    for batch_idx in range(n_batches):
        docs = files[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        tempdir = os.path.join(tempdirs, f"temp_{batch_idx}")
        os.mkdir(tempdir)
        for doc in docs:
            shutil.copy(os.path.join(data_path, doc), tempdir)
        temp_dirs.append(tempdir)
    print(f"found {n_batches} batches. temp dirs: {temp_dirs}")

    # Fire off subprocesses to process this batch in an azure VM. All of the work is
    # happening remotely, so it's ok to have a large number of processes in the pool.
    with Pool(len(temp_dirs)) as pool:
        all_results = pool.starmap(
            process_data_batch, zip(temp_dirs, list(range(len(temp_dirs))))
        )

    # Stitch together data batches
    return stitch_processed_data_batches(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="process data in azure in batches")
    parser.add_argument(
        "input_dir",
        type=str,
    )
    parser.add_argument("--batch_size", type=int, default=100, required=False)
    parser.add_argument(
        "--clear", action="store_true", help="clear tempdirs from prev runs"
    )
    # TODO customize image and cmd
    parser.add_argument("--image", type=str, default="", required=False)
    parser.add_argument("--cmd", type=str, default="", required=False)
    args = parser.parse_args()

    # Clear old data
    tempdir = args.input_dir.replace("/", "_")
    if args.clear:
        if os.path.exists(tempdir):
            shutil.rmtree(tempdir)

    # Process all data in the given input directory at the given batch size
    kgs, docs = process_all_data(
        args.input_dir, batch_size=args.batch_size, tempdirs=tempdir
    )

    # Save results locally
    with open(os.path.join(tempdir, "knowledge_graphs.pkl"), "wb") as f:
        pkl.dump((kgs, docs), f)
