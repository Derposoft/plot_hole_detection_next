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


def main():
    # Get a list of all NICs
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

    # Create a multiprocessing pool with the number of processes you desire
    pool = multiprocessing.Pool(
        processes=10
    )  # Adjust the number of processes as needed

    # print(nic_list)
    # Delete NICs and associated Public IPs in parallel
    pool.map(delete_nic_and_public_ip, nic_list)


if __name__ == "__main__":
    main()
