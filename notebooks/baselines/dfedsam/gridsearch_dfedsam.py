from ruamel.yaml import YAML
import itertools
import subprocess

# Define the hyperparameter space
# alhpa = 0.1, 0.5, for 3 trials

# user_with_data_fps = [
# "",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/fmnist300/a0.1/user_dataidx_map_0.10_0.dat",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/fmnist300/a0.2/user_dataidx_map_0.20_0.dat",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/fmnist300/a0.3/user_dataidx_map_0.30_0.dat",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/fmnist300/a0.4/user_dataidx_map_0.40_0.dat",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/mnist300/a0.5/user_dataidx_map_0.50_0.dat"
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/mnist300/a0.5/user_dataidx_map_0.50_0.dat",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/mnist300/a0.1/user_dataidx_map_0.10_0.dat"
# ]
user_with_data_fps = [
    f"/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/fmnist300/a0.5/user_dataidx_map_0.50_{i}.dat" for i in range(5)
]


# # Add IID fp
# user_with_data_fps.append("")

param_grid = {
    'lr': [0.01],
    "local_epochs": [5],
    "rho": [0.01],
    "adaptive": [True],
    "lr_decay": [0.95],
    "momentum": [0.99],
    "wd": [5.0e-4],
    "optimizer_batch_size": [32],
    'static': [False],
    'user_with_data': user_with_data_fps,
    'd': [5]
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    param_grid['lr'],
    param_grid['local_epochs'],
    param_grid['rho'],
    param_grid['adaptive'],
    param_grid['lr_decay'],
    param_grid['momentum'],
    param_grid['wd'],
    param_grid['optimizer_batch_size'],
    param_grid['static'],
    param_grid['user_with_data'],
    param_grid['d']
))

# Function to update the YAML file
def update_yaml(file_path, params):
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes if present in the YAML file

    with open(file_path, 'r') as file:
        config = yaml.load(file)

    config['lr'] = params[0]
    config['local_epochs'] = params[1]
    config['rho'] = params[2]
    config['adaptive'] = params[3]
    config['lr_decay'] = params[4]
    config['momentum'] = params[5]
    config['wd'] = params[6]
    config['optimizer_batch_size'] = params[7]
    config['static'] = params[8]
    config['user_with_data'] = params[9]
    config['d'] = params[10]

    with open(file_path, 'w') as file:
        yaml.dump(config, file)

yaml_file_path = "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/utils/baseline_configs/config_dfedsam.yaml"

# Loop through all combinations and run the model for each set
for params in param_combinations:
    update_yaml(yaml_file_path, params)
    
    # Run your model script that uses the YAML file
    # This could be a Python script executed from the terminal, or any other executable
    command = ['python', 'dfedsam.py', yaml_file_path]
    subprocess.run(command)
    
    print(f'Completed run with hyperparameters: {params}')