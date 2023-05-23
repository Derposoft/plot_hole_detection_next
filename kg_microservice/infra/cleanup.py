import multiprocessing
import subprocess

N_PROCS = 50


def delete_vm(vm_name):
    # Delete the VM
    subprocess.run(
        ["az", "vm", "delete", "--name", vm_name, "-g", "plot_hole_detection", "--yes"],
        check=True,
    )


def delete_vms():
    # Get a list of all VMs
    vm_list = subprocess.run(
        [
            "az",
            "vm",
            "list",
            "--query",
            "[].name",
            "--output",
            "tsv",
            "-g",
            "plot_hole_detection",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    # Create a multiprocessing pool with the number of processes you desire
    pool = multiprocessing.Pool(
        processes=N_PROCS
    )  # Adjust the number of processes as needed

    # Delete VMs in parallel
    pool.map(delete_vm, vm_list)


def delete_disk(disk_name):
    # Delete the disk
    subprocess.run(
        [
            "az",
            "disk",
            "delete",
            "--name",
            disk_name,
            "--yes",
            "-g",
            "plot_hole_detection",
        ],
        check=True,
    )


def delete_disks():
    # Get a list of all disks
    disk_list = subprocess.run(
        [
            "az",
            "disk",
            "list",
            "--query",
            "[].name",
            "--output",
            "tsv",
            "-g",
            "plot_hole_detection",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    # Create a multiprocessing pool with the number of processes you desire
    pool = multiprocessing.Pool(
        processes=N_PROCS
    )  # Adjust the number of processes as needed

    # Delete disks in parallel
    pool.map(delete_disk, disk_list)


def delete_nic_and_public_ip(nic_name):
    # Delete the NIC
    subprocess.run(
        [
            "az",
            "network",
            "nic",
            "delete",
            "-g",
            "plot_hole_detection",
            "--name",
            nic_name,
        ],
        check=True,
    )


def delete_public_ip(public_ip_name):
    # Delete the Public IP address
    subprocess.run(
        [
            "az",
            "network",
            "public-ip",
            "delete",
            "-g",
            "plot_hole_detection",
            "--name",
            public_ip_name,
        ],
        check=True,
    )


def delete_ips():
    # delete nics
    nic_list = subprocess.run(
        [
            "az",
            "network",
            "nic",
            "list",
            "-g",
            "plot_hole_detection",
            "--query",
            "[].name",
            "--output",
            "tsv",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pool = multiprocessing.Pool(processes=N_PROCS)
    pool.map(delete_nic_and_public_ip, nic_list)

    # delete public ips
    public_ip_list = subprocess.run(
        ["az", "network", "public-ip", "list", "--query", "[].name", "--output", "tsv"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pool = multiprocessing.Pool(processes=N_PROCS)
    pool.map(delete_public_ip, public_ip_list)


if __name__ == "__main__":
    delete_vms()
    delete_ips()
    delete_disks()
