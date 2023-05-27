import sys
from azure_process_data import create_azure_vm
import paramiko

train_image = "/subscriptions/5a5fec05-4c7d-4b5f-9142-bf8a1f62966d/resourceGroups/plot_hole_detection/providers/Microsoft.Compute/galleries/stanfordcorenlp/images/train/versions/2.0.0"

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 azure_rum_cmd.py vm_name [command to run on vm]")
        sys.exit()
    name = sys.argv[1]
    cmd = " ".join(sys.argv[2:])

    # Create new vm with given name
    host, username, password = create_azure_vm(name=name, image=train_image)
    port = 22

    # Connect to vm and start the command
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port, username, password)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    ssh.close()

    # Return stdin/stdout/stderr and exit
    print("Running!")
    print(f"Input: {stdin}")
    print(f"Output: {stdout}")
    print(f"Error: {stderr}")
