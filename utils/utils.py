import os
import pickle
import logging
import datetime
import numpy as np

import torch

from fedlearning.model import init_weights
from fedlearning import nn_registry
from fedlearning.evolve import WeightMod
import copy

def tensor_size_in_bytes(tensor):
    return tensor.nelement() * tensor.element_size()

def jac_size_in_bytes(package):
    size = 0.0
    for key in package.keys():
        size += tensor_size_in_bytes(package[key])
    return size

def init_logger(config):
    """Initialize a logger object. 
    """
    
    log_level = logging.INFO    
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # current_path = os.path.dirname(__file__)
    # current_path = os.path.dirname(current_path)
    # current_path = os.path.dirname(current_path)
    # current_path = os.path.join(current_path, config.log_file)
    current_path = config.log_file
    
    fh = logging.FileHandler(current_path)
    fh.setLevel(log_level)
    sh = logging.StreamHandler()
    sh.setLevel(log_level)
        
    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info("-"*80)
    
    return logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_record(config, model):
    record = {}
    # number of trainable parameters
    record["num_parameters"] = count_parameters(model)

    # put some config info into record
    record["batch_size"] = config.local_batch_size
    record["lr"] = config.lr
    record["taus"] = []

    # initialize data record 
    record["testing_accuracy"] = []
    record["loss"] = []
    record["rounds"] = config.rounds
    record["iid"] = True if config.iid else False
    record["model_accs_per_round"] = []
    record["model_num_training_rounds"] = []
    record["epoch"] = 0
    return record

def load_record(filepath):
    assert(os.path.exists(filepath))
    with open(filepath, 'rb') as file:
        record = pickle.load(file)
        return record

# def save_record(config, record):
#     current_path = os.path.dirname(__file__)
#     current_time = datetime.datetime.now()
#     current_time_str = datetime.datetime.strftime(current_time ,r'%d_%m_%H_%M')
#     file_name = config.record_dir.format(current_time_str)
#     parent_path = os.path.dirname(current_path)
#     with open(os.path.join(current_path, "records", file_name), "wb") as fp:
#         pickle.dump(record, fp)

def save_record(config, record):
    if os.path.isabs(config.record_dir):
        current_time = datetime.datetime.now()
        current_time_str = datetime.datetime.strftime(current_time ,r'%d_%m_%H_%M')
        file_path = config.record_dir.format(current_time_str)
    else:
        current_path = os.path.dirname(__file__)
        current_path = os.path.dirname(current_path)
        current_time = datetime.datetime.now()
        current_time_str = datetime.datetime.strftime(current_time ,r'%d_%m_%H_%M')
        file_name = config.record_dir.format(current_time_str)
        # parent_path = os.path.dirname(current_path)
        # record_dir = os.path.join(current_path, "records")
        os.makedirs(current_path, exist_ok=True)
        file_path = os.path.join(current_path, file_name)
    with open(file_path, "wb") as fp:
        pickle.dump(record, fp)

def parse_model(config):
    if config.model in nn_registry.keys():
        return nn_registry[config.model]

    if "cifar" in config.train_data_dir:
        return nn_registry["cifar_mlp"]
    elif "fmnist" in config.train_data_dir:
        return nn_registry["fmnist_mlp"]
    else:
        return nn_registry["mnist_mlp"]

def parse_dataset_type(config):
    if "fmnist" in config.train_data_dir:
        type_ = "fmnist"
    elif "mnist" in config.train_data_dir:
        type_ = "mnist"
    elif "cifar" in config.train_data_dir:
        type_ = "cifar"
    
    return type_

def init_model(config, logger):
    # initialize the model
    sample_size = config.datapoint_size[0] * config.datapoint_size[1] * config.channels
    if config.model == "mlp":
        full_model = nn_registry[config.model](in_dims=sample_size, in_channels=config.channels, out_dims=config.label_size)
    elif config.model == "cnn":
        full_model = nn_registry[config.model](in_dims=[config.datapoint_size[0], config.datapoint_size[1]], in_channels=config.channels, out_dims=config.label_size)
    else:
        raise ValueError("Model type not supported.")
    
    full_model.apply(init_weights)

    if os.path.exists(config.full_weight_dir):
        logger.info("--- Load pre-trained full precision model. ---")
        state_dict = torch.load(config.full_weight_dir)
        full_model.load_state_dict(state_dict)

    full_model.to(config.device)

    return full_model

def average_neighbor_weights(client_id, neighbor_ids, model_dict):
    # Average the weights of the models in the cluster
    weight_dict = copy.deepcopy(model_dict[client_id].state_dict())
    weight_aggregator = WeightMod(weight_dict)
    for user_id in neighbor_ids:
        weight_aggregator.add(copy.deepcopy(model_dict[user_id].state_dict()))
    # Add one for the client itself
    weight_aggregator.mul(1.0/ (len(neighbor_ids)+1) )
    return weight_aggregator.state_dict()