import yaml
import os
import numpy as onp

def load_config(filename=None):
    """Load configurations of yaml file"""
    current_path = os.path.dirname(__file__)
    if filename is None:
        raise ValueError("No filename provided")
        #filename = "config.yaml"
    print("Creating config from filepath: ", filename)
    if os.path.isabs(filename):
        with open(filename, "r") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)
    else: 
        with open(os.path.join(current_path, filename), "r") as fp:
            config = yaml.load(fp, Loader=yaml.FullLoader)

    # Empty class for yaml loading
    class cfg: pass
    
    for key in config:
        setattr(cfg, key, config[key])
    
    if not hasattr(cfg, "model"):
        cfg.model = ""

    return cfg
