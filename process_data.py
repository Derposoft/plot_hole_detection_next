import os
import sys
import data.utils as utils

data_path = sys.argv[1]
n_continuity_errors = 1
if len(sys.argv) > 2:
    n_continuity_errors = int(sys.argv[2])

n_stories = len([x for x in os.listdir(data_path) if x.endswith(".txt")])
utils.generate_data(
	batch_size = 8,
    n_stories = n_stories,
    n_synth = 1,
	data_path = data_path,
	cache_path = data_path,
	get_kgs = True,
    n_continuity_errors = n_continuity_errors,
)
