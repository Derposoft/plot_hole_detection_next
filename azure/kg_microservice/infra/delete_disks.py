import multiprocessing
import subprocess


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


if __name__ == "__main__":
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
        processes=100
    )  # Adjust the number of processes as needed

    # Delete disks in parallel
    pool.map(delete_disk, disk_list)
