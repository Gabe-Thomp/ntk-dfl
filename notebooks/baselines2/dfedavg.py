import sys
directory_path = "../../../"
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
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas

# Necessary for optimization with pytorch
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

def train_on_client_data(user_id, global_model, dataset, config, logger, 
local_update_steps, local_bs, lr, loss_fn = "ce", verbose=False):
    # Create a copy of the global model to be used for training
    user_model = copy.deepcopy(global_model)
    
    # Get data corresponding to a certain user
    user_resource = assign_user_resource(config, user_id, 
                        dataset["train_data"], dataset["user_with_data"])
    
    # Define the optimizer
    optimizer = optim.SGD(user_model.parameters(), lr=lr)
    
    # Define the dataset
    np_dataset = NumpyDataset(user_resource["images"], user_resource["labels"], transform=numpy_to_tensor_transform)
    
    # Define the dataloader
    user_data_loader = DataLoader(np_dataset, batch_size=local_bs, shuffle=True)

    # Define the loss function
    if loss_fn == "ce": 
        criterion = nn.CrossEntropyLoss()
    else: 
        raise ValueError("Loss function not implemented")

    for local_epoch in range(local_update_steps):
         # Iterate over the user's data
        for batch_idx, (data, target) in enumerate(user_data_loader):
            data, target = data.to(config.device), target.to(config.device)
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = user_model(data)
            
            # Compute the loss
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update the model parameters
            optimizer.step()
    
    return user_model

def average_neighbor_weights(client_id, neighbor_ids, model_dict):
    # Average the weights of the models in the cluster
    weight_dict = copy.deepcopy(model_dict[client_id].state_dict())
    weight_aggregator = WeightMod(weight_dict)
    for user_id in neighbor_ids:
        weight_aggregator.add(copy.deepcopy(model_dict[user_id].state_dict()))
    # Add one for the client itself
    weight_aggregator.mul(1.0/ (len(neighbor_ids)+1) )
    return weight_aggregator.state_dict()

def perform_fedavg(config, logger, record, loaded_record=False):
    # Create user_ids
    user_ids = np.arange(0, config.users)
    # load the dataset
    # dataset object is a dictionary with keys: train_data, test_data, user_with_data
    # user_with_data is a dictionary with keys: userID:sampleID
    # For example, in the IID setting ID's are just assigned like 0, 1, 2, 3, ...
    dataset = assign_user_data(config, logger)
    test_images = torch.from_numpy(dataset["test_data"]["images"]).to(config.device)
    test_labels = torch.from_numpy(dataset["test_data"]["labels"]).to(config.device)
    
    # Initialize the model
    if loaded_record == False: 
        # Init model if none provided
        global_model = init_model(config, logger)
    else:
        # Load model if provided
        global_model = init_model(config, logger)
        global_model.load_state_dict(record["global_model"])

    for comm_round in range(config.rounds):
        # Empty model dict to store the client updated models
        temp_model_dict = {}

        # Select C fraction of clients randomly
        participating_client_ids = np.random.choice(user_ids, int(config.part_rate * config.users), replace=False)
        
        # Train on all participating clients
        for client_id in participating_client_ids:
            # Train on the client's data
            user_model = train_on_client_data(client_id, global_model, dataset, config, logger, 
                                            config.local_update_steps, config.sgd_batch_size, config.lr, 
                                            loss_fn = config.loss, verbose=config.verbose)
            temp_model_dict[client_id] = user_model
        
        # Average the deviated weights
        averaged_state_dict = average_neighbor_weights(participating_client_ids[0], participating_client_ids[1:], temp_model_dict)
        
        # Load the averaged weights to the global model
        global_model.load_state_dict(averaged_state_dict)

        # Test the global model
        output = global_model(test_images)
        loss = nn.CrossEntropyLoss()(output, test_labels)
        acc = accuracy_with_output(output, test_labels)
        if comm_round % 10 == 0: logger.info(f"Round {comm_round}: Test Loss: {loss.item()}, Accuracy: {acc}")

        # Record the results
        record["loss"].append(loss.item())
        record["testing_accuracy"].append(acc)
        record["epoch"] += 1

    # Save the record
    record["global_model"] = global_model.state_dict()
    record["fedavg_hyperparameters"] = {"learning_rate": config.lr, "local_update_steps": config.local_update_steps,
                                        "sgd_batch_size": config.sgd_batch_size, "participation_rate": config.part_rate}  
    record["user_with_data"] = config.user_with_data
    
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
    perform_fedavg(config, logger, record, loaded_record = loaded_record)
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
