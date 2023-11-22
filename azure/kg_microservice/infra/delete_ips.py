import multiprocessing
import subprocess


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


if __name__ == "__main__":
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
    pool = multiprocessing.Pool(processes=50)
    pool.map(delete_nic_and_public_ip, nic_list)

    # delete public ips
    public_ip_list = subprocess.run(
        ["az", "network", "public-ip", "list", "--query", "[].name", "--output", "tsv"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    pool = multiprocessing.Pool(processes=50)
    pool.map(delete_public_ip, public_ip_list)
