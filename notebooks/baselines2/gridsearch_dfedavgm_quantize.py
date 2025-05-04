from ruamel.yaml import YAML
import itertools
import subprocess

# Define the hyperparameter space
# alhpa = 0.1, 0.5, for 3 trials
# user_with_data_fps = ["/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/mnist300/a0.05/user_dataidx_map_0.05_0.dat", 
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/mnist300/a0.1/user_dataidx_map_0.10_0.dat",
# "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/data/user_with_data/mnist300/a0.5/user_dataidx_map_0.50_0.dat"]
# # Add IID fp
# user_with_data_fps.append("")

param_grid = {
    "quant_level": [256],
}

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(
    param_grid['quant_level'],
))

# Function to update the YAML file
def update_yaml(file_path, params):
    yaml = YAML()
    yaml.preserve_quotes = True  # Preserve quotes if present in the YAML file

    with open(file_path, 'r') as file:
        config = yaml.load(file)

    config['quant_level'] = params[0]


    with open(file_path, 'w') as file:
        yaml.dump(config, file)

yaml_file_path = "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/utils/baseline_configs/config_dfedavgm_quantize.yaml"

# Loop through all combinations and run the model for each set
for params in param_combinations:
    update_yaml(yaml_file_path, params)
    
    # Run your model script that uses the YAML file
    # This could be a Python script executed from the terminal, or any other executable
    command = ['python', 'dfedavgm_quantize.py', yaml_file_path]
    subprocess.run(command)
    
    print(f'Completed run with hyperparameters: {params}')