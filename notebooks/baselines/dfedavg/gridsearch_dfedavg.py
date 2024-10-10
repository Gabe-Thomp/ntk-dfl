from ruamel.yaml import YAML
import itertools
import subprocess

# Define the hyperparameter space
# alhpa = 0.1, 0.5, for 3 trials
user_with_data_fps = [f"../../../data/user_with_data/fmnist300/a0.{i}/user_dataidx_map_0.{i}0_{j}.dat" for i in [1,5] for j in range(0, 3)]
# Add IID fp
# user_with_data_fps.append("")

param_grid = {
    'lr': [0.1],
    'sgd_batch_size': [25],
    'local_update_steps': [20],
    'momentum': [0],
    "static": [False],
    'user_with_data': user_with_data_fps
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    param_grid['lr'],
    param_grid['sgd_batch_size'],
    param_grid['local_update_steps'],
    param_grid['momentum'],
    param_grid['static'],
    param_grid['user_with_data']
))

# Function to update the YAML file
def update_yaml(file_path, params):
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes if present in the YAML file

    with open(file_path, 'r') as file:
        config = yaml.load(file)

    config['lr'] = params[0]
    config['sgd_batch_size'] = params[1]
    config['local_update_steps'] = params[2]
    config['momentum'] = params[3]
    config['static'] = params[4]
    config['user_with_data'] = params[5]

    with open(file_path, 'w') as file:
        yaml.dump(config, file)

# Note: replace this with your own path to the config file
yaml_file_path = "ADD_YOUR_PATH/ntk_dfl/utils/baseline_configs/config_dfedavg.yaml"

# Loop through all combinations and run the model for each set
for params in param_combinations:
    update_yaml(yaml_file_path, params)
    
    # Run your model script that uses the YAML file
    # This could be a Python script executed from the terminal, or any other executable
    command = ['python', '../dfedavgm/dfedavgm.py', yaml_file_path]
    subprocess.run(command)
    
    print(f'Completed run with hyperparameters: {params}')