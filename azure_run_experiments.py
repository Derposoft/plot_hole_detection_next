"""
runs all of our experiments
"""
import os
from multiprocessing import Pool

experiments = [
    # 1 error experiments
    "python azure_run_cmd.py 1_err-lstm 'nohup python3 -u train.py --model lstm --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 1_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 1_err-textcnn 'nohup python3 -u train.py --model textcnn --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 1_err-bert_kg 'nohup python3 -u train.py --model bert_kg --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 1_err-get 'nohup python3 -u train.py --model get --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 1_err-mac 'nohup python3 -u train.py --model mac --n_continuity_errors 1 &'",
    # 2 error experiments
    "python azure_run_cmd.py 2_err-lstm 'nohup python3 -u train.py --model lstm --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 2_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 2_err-textcnn 'nohup python3 -u train.py --model textcnn --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 2_err-bert_kg 'nohup python3 -u train.py --model bert_kg --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 2_err-get 'nohup python3 -u train.py --model get --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 2_err-mac 'nohup python3 -u train.py --model mac --n_continuity_errors 2 &'",
    # 5 error experiments
    "python azure_run_cmd.py 5_err-lstm 'nohup python3 -u train.py --model lstm --n_continuity_errors 5 &'",
    "python azure_run_cmd.py 5_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 5 &'",
    "python azure_run_cmd.py 5_err-textcnn 'nohup python3 -u train.py --model textcnn --n_continuity_errors 5 &'",
    "python azure_run_cmd.py 5_err-bert_kg 'nohup python3 -u train.py --model bert_kg --n_continuity_errors 5 &'",
    "python azure_run_cmd.py 5_err-get 'nohup python3 -u train.py --model get --n_continuity_errors 5 &'",
    "python azure_run_cmd.py 5_err-mac 'nohup python3 -u train.py --model mac --n_continuity_errors 5 &'",
]


def execute_command(cmd):
    os.system(cmd)


if __name__ == "__main__":
    with Pool(os.cpu_count()) as p:
        p.map(execute_command, experiments)
