"""
runs all of our experiments
"""
import os
from multiprocessing import Pool

declare_experiments = [
    "python azure_run_cmd.py 1_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 2_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 5_err-declare 'nohup python3 -u train.py --model declare --n_continuity_errors 5 &'",
]

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
    # Alt GNN ablation experiments
    "python azure_run_cmd.py 1_err-bert_gcn 'nohup python3 -u train.py --model bert_kg --gnn_type gcn --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 1_err-bert 'nohup python3 -u train.py --model bert --n_continuity_errors 1 &'",
    "python azure_run_cmd.py 2_err-bert_gcn 'nohup python3 -u train.py --model bert_kg --gnn_type gcn --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 2_err-bert 'nohup python3 -u train.py --model bert --n_continuity_errors 2 &'",
    "python azure_run_cmd.py 5_err-bert_gcn 'nohup python3 -u train.py --model bert_kg --gnn_type gcn --n_continuity_errors 5 &'",
    "python azure_run_cmd.py 5_err-bert 'nohup python3 -u train.py --model bert --n_continuity_errors 5 &'",
]


# Select which set of experiments to run
experiments = declare_experiments


def execute_command(cmd):
    os.system(cmd)


if __name__ == "__main__":
    N_PROCS = 2 * os.cpu_count()  # eh why not
    with Pool(N_PROCS) as p:
        p.map(execute_command, experiments)
