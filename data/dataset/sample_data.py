"""
How to use this file:

python train.py --gen_data_only --n_stories 1000 --n_synth 10 --n_continuity_errors 1
python sample_data.py ../synthetic/train/ 1_error/train/ 8000
python sample_data.py ../synthetic/test/ 1_error/test/ 2000

Yes, I know I could've juts done it correctly in generate_synthetic_data.py. But I didn't,
so here we are.
"""

import os
import sys
import shutil
import numpy as np

if len(sys.argv) != 4:
    print(f"{len(sys.argv)} args received: {sys.argv}")
    print("usage: python3 sample_data.py path/to/files output/path n_samples")
    sys.exit()

input_path = sys.argv[1]
output_path = sys.argv[2]
n_samples = int(sys.argv[3])


input_files = [f for f in os.listdir(input_path) if f.endswith(".txt")]
sample_idxs = np.random.choice(len(input_files), n_samples, replace=False)
sampled_files = [input_files[i] for i in sample_idxs]


for file in sampled_files:
    shutil.copy(os.path.join(input_path, file), os.path.join(output_path, file))
