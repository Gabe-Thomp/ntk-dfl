from ruamel.yaml import YAML
import itertools
import subprocess

# Define the hyperparameter space
# Replace this fp list with the list of user_with_data files you want to use
user_with_data_fps = ["../../../data/user_with_data/emnist/digits/user_dataidx_map_0.dat"]

param_grid = {
    'lr': [0.1],
    'sgd_batch_size': [10],
    'local_update_steps': [10],
    'frac': [0.01666666],
    'user_with_data': user_with_data_fps
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    param_grid['lr'],
    param_grid['sgd_batch_size'],
    param_grid['local_update_steps'],
    param_grid['frac'],
    param_grid['user_with_data']
))

# Function to update the YAML file
def update_yaml(file_path, params):
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes if present in the YAML file

    with open(file_path, 'r') as file:
        config = yaml.load(file)

    config['lr'] = params[0]
    config['batch_size'] = params[1]
    config['epochs'] = params[2]
    config['frac'] = params[3]
    config['user_with_data'] = params[4]

    with open(file_path, 'w') as file:
        yaml.dump(config, file)

# Note: replace this with your own path to the config file
yaml_file_path = "ADD_YOUR_PATH/Deep_Learning/ntk_dfl/utils/baseline_configs/config_dispfl.yaml"

# Loop through all combinations and run the model for each set
for params in param_combinations:
    update_yaml(yaml_file_path, params)
    
    # Run your model script that uses the YAML file
    # This could be a Python script executed from the terminal, or any other executable
    dispfl_main_fp = "./DisPFL/fedml_experiments/standalone/DisPFL/main_dispfl.py"
    command = ['python', dispfl_main_fp, "--config", yaml_file_path]
    subprocess.run(command)
    
    print(f'Completed run with hyperparameters: {params}')