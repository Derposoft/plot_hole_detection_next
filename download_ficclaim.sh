# Download dataset from google drive
python3 -m pip install gdown
gdown --fuzzy https://drive.google.com/file/d/12a0cS6aAgs5Expjat_wpf4qo1tUj8W1h/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1I0q0DN5KAGts3TcwcDunrjZDNM9KUI9V/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1KGpYoqdAL46-FQyegUWrgzxkw5YkSf5E/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1SGX3NMi8tVWtAvJ2rSR9CaIxukeCNi7Z/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1gJl_TpMmFstbxug1HilF5-fpLsjSorsS/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1y09wt6Ys1VpvmHUeIYraaDwilUK9vrSV/view?usp=sharing

# Expand the compressed dataset so that we can run training on it
python3 expand_dataset.py test_1_error.pkl
python3 expand_dataset.py test_2_error.pkl
python3 expand_dataset.py test_5_error.pkl
python3 expand_dataset.py train_1_error.pkl
python3 expand_dataset.py train_2_error.pkl
python3 expand_dataset.py train_5_error.pkl

# Move data into right folder
mv test*.pkl FicClaim/test
mv train*.pkl FicClaim/train
