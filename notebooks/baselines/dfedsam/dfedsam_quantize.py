import sys
directory_paths = ["../../../", "/home/gathomp3/Deep_Learning/NeuralTangent/ntk-fed/notebooks/baselines/dfedsam/DP-FedSAM"]

for directory_path in directory_paths:
    if directory_path not in sys.path:
        # Add the directory to sys.path
        sys.path.append(directory_path)

import copy
import time
import time
import numpy as np
import argparse
import yaml
import networkx as nx
import matplotlib.pyplot as plt

from torch.utils import data
from torch import optim

from utils.utils import *
from utils import load_config
from utils.validate import *

from fedlearning.topology import *
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas
from fedlearning.quantizer import DFedAvgQuantizer

from fedml_api.dpfedsam.sam import SAM

class NumpyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        """
        Args:
            data (numpy array): Array of data samples.
            targets (numpy array): Array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, target

def numpy_to_tensor_transform(data):
    return torch.from_numpy(data)

def train_client_dfedsam(user_model, user_id, dataset, config, logger, loss_fn, 
    local_epochs,
    comm_round,
    rho,
    adaptive,
    lr,
    lr_decay,
    momentum,
    wd,
    optimizer_batch_size
    ): 
    # Get data corresponding to a certain user
    user_resource = assign_user_resource(config, user_id, 
                    dataset["train_data"], dataset["user_with_data"])

    # Define cross-entropy criterion
    loss_fn_pytorch = nn.CrossEntropyLoss()

    # Define the SAM optimizer
    base_optimizer = torch.optim.SGD
    optimizer = SAM(user_model.parameters(), base_optimizer, rho=rho, adaptive=adaptive, 
        lr=lr* (lr_decay**comm_round), momentum=momentum, weight_decay=wd)

    # Dataset stuff
    np_dataset = NumpyDataset(user_resource["images"], user_resource["labels"], transform=numpy_to_tensor_transform)
    user_data_loader = DataLoader(np_dataset, batch_size=optimizer_batch_size, shuffle=True)

    # Doing local_epochs number of local training rounds
    for epoch in range(local_epochs):
        # Iterate over the user's data
        epoch_loss, epoch_acc = [], []
        for batch_idx, (x, labels) in enumerate(user_data_loader):
            x, labels = x.to(config.device), labels.to(config.device)

            # From the SAM codebase
            # first forward-backward step
            pred = user_model(x)
            
            # Don't need enable_running_stats due to no batchnorm or any moving averages
            # enable_running_stats(model)
            # log_probs = model.forward(x)
            # loss = loss_fn_pytorch(user_model(x), labels.long())
            loss = loss_fn_pytorch(user_model(x), labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            # Don't need disable_running_stats due to no batchnorm or any moving averages
            # disable_running_stats(model)
            # loss_fn_pytorch(user_model(x), labels.long()).backward()
            loss_fn_pytorch(user_model(x), labels).backward()
            optimizer.second_step(zero_grad=True)
            
            epoch_loss.append(loss.item())
            
        # if config.verbose: 
        #     print('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
        #     user_id, epoch, sum(epoch_loss) / len(epoch_loss)))

def perform_dfedsam(config, logger, record, loaded_record=False):
    # Create user_ids
    user_ids = np.arange(0, config.users)
    # load the dataset
    # dataset object is a dictionary with keys: train_data, test_data, user_with_data
    # user_with_data is a dictionary with keys: userID:sampleID
    # For example, in the IID setting ID's are just assigned like 0, 1, 2, 3, ...
    dataset = assign_user_data(config, logger)
    test_images = torch.from_numpy(dataset["test_data"]["images"]).to(config.device)
    test_labels = torch.from_numpy(dataset["test_data"]["labels"]).to(config.device)

    loaded_record = False
    # Create a dictionary of models for each user
    # Same initialization for all users
    # If record/model_dict is passed, continue training from where it left off
    if loaded_record == True:
        model_dict = record["models"]
        model_comm_record = record["model_comm_record"]
    else:
        if config.same_init:
            model = init_model(config, logger)
            model_dict = {model_id: copy.deepcopy(model) for model_id in user_ids}
        else:
            model_dict = {model_id: init_model(config, logger) for model_id in user_ids}
        # Initialize model communication record
        model_comm_record = {model_id: [] for model_id in user_ids}

    def quantize_weights(state_dict, b, s=None, quantizer="dfedavg", mode="copy"):
        """
        Quantize the jacobian of the model using the quantizer in DFedAvg paper
        Args:

        state_dict (dict): The state dictionary of the model
        b: The number of bits to quantize to
        s: Step size for quantizer. If none, s is determined by the range of the weights
        """
        if quantizer == "dfedavg":
            quantizer = DFedAvgQuantizer(b=b, s=s)
        else: 
            raise ValueError("Quantizer not supported")
            
        wmod = WeightMod(state_dict, mode=mode)
        bits = wmod.apply_quant(quantizer)
        return wmod.state_dict(), bits

    # Configure quantization parameters
    if np.log2(config.quant_level).is_integer():
        quantization_bit_parameter = int(np.log2(config.quant_level))
    else: 
        raise ValueError("Quantization level must be a power of 2")

    # Log inital loss and accuracy
    if record["epoch"] == 0:
        logger.info("Logging initial loss, acc")
        client_losses = []
        client_accs = []
        for client_id in user_ids:
            # Evaluate the client's model on the slice of the training data corresponding to the client's data
            user_images = torch.from_numpy(dataset["train_data"]["images"][dataset["user_with_data"][client_id]]).to(config.device)
            user_labels = torch.from_numpy(dataset["train_data"]["labels"][dataset["user_with_data"][client_id]]).to(config.device)
            
            # Get model outputs
            output_on_own_data = model_dict[client_id](user_images)
            output_on_test_set = model_dict[client_id](test_images)
            
            # Get losses/accs, and append to list
            loss = loss_with_output(output_on_own_data, user_labels, config.loss)
            acc = accuracy_with_output(output_on_test_set, test_labels)
            
            client_losses.append(loss)
            client_accs.append(acc)
            if config.verbose: logger.info("client {:d} loss {:.4f} acc {:.4f}".format(client_id, loss, acc))

        record["loss"].append(client_losses)
        record["testing_accuracy"].append(client_accs)

    assert(type(config.static) == bool)
    if config.static == True:
        logger.info("Static setting: All clients connected on static topology")
        # Create the graph for this round
        if config.topology == "random":
            G = create_random_graph(config.users, config.p, config.graph_name)
        elif config.topology == "ring":
            G = create_ring_graph(config.users, config.graph_name)
        elif config.topology == "regular":
            if config.p is not None:
                raise ValueError("Regular graph requires d, not p")
            elif config.d is None:
                raise ValueError("Regular graph requires d")
            G = create_regular_graph(config.users, config.d, config.graph_name)
            if config.verbose: logger.info(f"Creating regular graph with d={config.d}")
        else: 
            raise ValueError("Invalid topology: {}".format(config.topology))

    for comm_round in range(record["epoch"],record["epoch"]+config.rounds):
        logger.info(f"Comm Round: {comm_round}")
        client_losses = []
        client_accs = []

        # Append a new, empty communication count for each client
        if model_comm_record is not None:
            for value in model_comm_record.values():
                value.append(0)
        else: 
            raise ValueError("model_comm_record is None")
        
        if config.static == False:
            # Create the graph for this round
            if config.topology == "random":
                G = create_random_graph(config.users, config.p, config.graph_name)
            elif config.topology == "ring":
                G = create_ring_graph(config.users, config.graph_name)
            elif config.topology == "regular":
                if config.p is not None:
                    raise ValueError("Regular graph requires d, not p")
                elif config.d is None:
                    raise ValueError("Regular graph requires d")
                G = create_regular_graph(config.users, config.d, config.graph_name)
                if config.verbose: logger.info(f"Creating regular graph with d={config.d}")
            else: 
                raise ValueError("Invalid topology: {}".format(config.topology))
    
        # All clients perform multiple rounds of local training
        for client_id in user_ids:
            # SGD w/ momentum
            loss_fn_pytorch = nn.CrossEntropyLoss()
            train_client_dfedsam(model_dict[client_id], client_id, dataset, config, logger, loss_fn_pytorch, 
            comm_round = comm_round,
            # Dfedsam parameters
            local_epochs = config.local_epochs,
            rho = config.rho,
            adaptive = config.adaptive,
            lr = config.lr,
            lr_decay = config.lr_decay,
            momentum = config.momentum,
            wd = config.wd,
            optimizer_batch_size = config.optimizer_batch_size)
        
        # All clients quantize their weights before sending to neighbors
        for client_id in user_ids:
            if config.verbose: logger.info(f"Quantizing model for client {client_id}")
            temp_dict, temp_bits = quantize_weights(model_dict[client_id].state_dict(), b=quantization_bit_parameter)
            model_dict[client_id].load_state_dict(temp_dict)
            # Convert to MB, then add to model_comm_record
            if model_comm_record is not None:
                neighbors = list(G.neighbors(client_id))
                num_neighbors = len(neighbors)
                # Add the communication cost to the record
                # Multiply by number of neighbors who the model is sent to
                model_comm_record[client_id][-1] += num_neighbors*temp_bits/8e6
            else: 
                raise ValueError("model_comm_record is None")

        # All clients average with neighbors
        # Must find the avged weights then load weights to mimic synchronous averaging
        new_avged_weights = {}
        # Get new weights for all clients
        for client_id in user_ids:
            neighbors = list(G.neighbors(client_id))
            new_avged_weights[client_id] = average_neighbor_weights(client_id, neighbors, model_dict)
        # Load new weights for all clients
        for client_id in user_ids:
            model_dict[client_id].load_state_dict(new_avged_weights[client_id])
        del new_avged_weights
        
        # Now, test individual client accs
        for client_id in user_ids:
            # Get client accuracy
            output_on_test_set = model_dict[client_id](test_images)
            acc = accuracy_with_output(output_on_test_set, test_labels)
            client_accs.append(acc)
        
        # Test the global, aggregated model
        # Note: Weighted averaging is unnecessary since all clients have the same number of samples
        
        # Init model to load aggregated state dict
        temp_global_model = init_model(config, logger)
        temp_global_model.load_state_dict(average_neighbor_weights(0, user_ids[1:], model_dict))
        
        global_output = temp_global_model(test_images)
        global_loss = nn.CrossEntropyLoss()(global_output, test_labels)
        global_acc = accuracy_with_output(global_output, test_labels)
        
        if comm_round % 5 == 0: 
            logger.info(f"Round {comm_round}: Test Loss: {global_loss.item()}, Avg Client Acc: {np.mean(client_accs)}, Agg Acc: {global_acc}")

        # Record the results
        record["testing_accuracy"].append(client_accs)
        
        if 'aggregated_accs' in record:
            record['aggregated_accs'].append(global_acc)
        else:
            record['aggregated_accs'] = [global_acc]

        record["epoch"] += 1

    # Save items to record
    record["models"] = model_dict
    record["dfedsam_hyperparameters"] = {"local_epochs": config.local_epochs,
            "rho": config.rho,
            "adaptive": config.adaptive,
            "lr": config.lr,
            "lr_decay": config.lr_decay,
            "momentum": config.momentum,
            "wd": config.wd,
            "optimizer_batch_size": config.optimizer_batch_size,
            "quanit_level": config.quant_level}
    record["user_with_data"] = config.user_with_data
    record["topology"] = config.topology
    record["p"] = config.p
    record["d"] = config.d
    record["model_comm_record"] = model_comm_record
    
    # Save the record
    save_record(config, record)

def main(config_file):
    config = load_config(config_file)

    logger = init_logger(config)
    logger.info("Loaded configuration from {}".format(config_file))
    logger.info("Dataset path: {}".format(config.train_data_dir))
    
    # Log if IID
    if config.user_with_data == "":
        logger.info("IID Dataset")
    else:
        logger.info(f"Using \"{config.user_with_data}\" premade Non-IID dataset")

    if config.record_path is not None:
        record = load_record(config.record_path)
        logger.info("Loaded record from {}".format(config.record_path))
        loaded_record = True
    else:
        # Define a model to extract number of parameters for record
        model = init_model(config, logger)
        record = init_record(config, model)
        loaded_record = False

    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

     # All work is done in the perform_fedavg function
    start = time.time()
    perform_dfedsam(config, logger, record, loaded_record = loaded_record)
    end = time.time()
    # Log the time taken
    logger.info("{:.3f} mins has elapsed.".format((end-start)/60))

if __name__ == "__main__":
    print("Activating")
    parser = argparse.ArgumentParser(description="Parse particular config file for federated learning trial")
    parser.add_argument('config_file', type=str, help="The path to the configuration file.")
    args = parser.parse_args()
    
    print("Config file: ", args.config_file)
    main(args.config_file)
