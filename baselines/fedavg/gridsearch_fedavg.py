from ruamel.yaml import YAML
import itertools
import subprocess

# Define the hyperparameter space
param_grid = {
    'lr': [0.001, 0.01, 0.1],
    'sgd_batch_size': [10, 20, 50, 100, 200],
    'local_update_steps': [1, 5, 10, 20]
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    param_grid['lr'],
    param_grid['sgd_batch_size'],
    param_grid['local_update_steps']
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

    with open(file_path, 'w') as file:
        yaml.dump(config, file)

yaml_file_path = "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/utils/baseline_configs/config_fedavg.yaml"

# Loop through all combinations and run the model for each set
for params in param_combinations:
    update_yaml(yaml_file_path, params)
    
    # Run your model script that uses the YAML file
    # This could be a Python script executed from the terminal, or any other executable
    command = ['python', 'fedavg.py', yaml_file_path]
    subprocess.run(command)
    
    print(f'Completed run with hyperparameters: {params}')