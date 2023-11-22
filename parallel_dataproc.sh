python3 parallel_dataproc_batch_stitch_carc.py data/synthetic/train --n_cont_error 1
cd ../jobs/gendata
job_id=$(sbatch job.gendata.graphonly.giggaparallel | awk '{print $4}')
scontrol wait "$job_id"
python3 parallel_dataproc_batch_stitch_carc.py data/synthetic/train --fix
