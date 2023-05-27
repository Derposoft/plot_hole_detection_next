import sys
import data.utils as utils

data_path = sys.argv[1] # "dataset/"

utils.generate_data(
	batch_size = 8,
	data_path = data_path,
	cache_path = ".",
	get_kgs = True,
)
