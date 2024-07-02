import sys
directory_path = "../"
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

from utils.utils import *
from utils import load_config
from utils.validate import *
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

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

def self_train(user_model, user_id, dataset, config, logger, loss_fn, batch_size=32, epochs=1, lr = 0.001, verbose=False): 
    # Get data corresponding to a certain user
    user_resource = assign_user_resource(config, user_id, 
                        dataset["train_data"], dataset["user_with_data"])
    
    # Define the optimizer
    optimizer = optim.SGD(user_model.parameters(), lr=lr)
    dataset = NumpyDataset(user_resource["images"], user_resource["labels"], transform=numpy_to_tensor_transform)

    user_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
   
    for epoch in range(epochs):
        # Iterate over the user's data
        for batch_idx, (data, target) in enumerate(user_data_loader):
            data, target = data.to(config.device), target.to(config.device)
            # Clear the gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = user_model(data)
            
            # Compute the loss
            loss = loss_fn(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update the model parameters
            optimizer.step()

            if batch_idx % 100 == 0:
                if verbose: logger.info(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(user_data_loader.dataset)} ({100. * batch_idx / len(user_data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    if verbose: print()

def create_random_graph(n, p, graph_name=None):
    # Generate the graph
    G = nx.erdos_renyi_graph(n, p)

    if graph_name != None:
        # Draw the graph
        nx.draw(G, with_labels=True)
        plt.savefig(graph_name)
    return G

def create_ring_graph(n, graph_name=None):
    G = nx.cycle_graph(n)
    if graph_name != None:
        # Draw the graph
        nx.draw(G, with_labels=True)
        plt.savefig(graph_name)
    return G

def average_neighbor_weights(client_id, neighbor_ids, model_dict):
    # Average the weights of the models in the cluster
    weight_dict = copy.deepcopy(model_dict[client_id].state_dict())
    weight_aggregator = WeightMod(weight_dict)
    for user_id in neighbor_ids:
        weight_aggregator.add(copy.deepcopy(model_dict[user_id].state_dict()))
    # Add one for the client itself
    weight_aggregator.mul(1.0/ (len(neighbor_ids)+1) )
    return weight_aggregator.state_dict()


def load_and_deload_neighbor_weights(neighbor_ids, model_dict, avg_weight_dict):
    # Save the weights of the neighbors
    older_weight_dicts = [copy.deepcopy(model_dict[user_id].state_dict()) for user_id in neighbor_ids]
    # Load the average weights
    for user_id in neighbor_ids:
        model_dict[user_id].load_state_dict(avg_weight_dict)
    return older_weight_dicts

def reload_neighbor_weights(neighbor_ids, model_dict, old_weight_dicts):
    for i, user_id in enumerate(neighbor_ids):
        model_dict[user_id].load_state_dict(old_weight_dicts[i])

def train(config, logger, record, loaded_record):
    # Create user_ids
    user_ids = np.arange(0, config.users)

    # load the dataset
    # dataset object is a dictionary with keys: train_data, test_data, user_with_data
    # user_with_data is a dictionary with keys: userID:sampleID
    # For example, in the IID setting ID's are just assigned like 0, 1, 2, 3, ...
    dataset = assign_user_data(config, logger)
    test_images = torch.from_numpy(dataset["test_data"]["images"]).to(config.device)
    test_labels = torch.from_numpy(dataset["test_data"]["labels"]).to(config.device)

    # tau candidates 
    taus = np.array(config.taus)
    
    # Create a dictionary of models for each user
    # Same initialization for all users
    # If record/model_dict is passed, continue training from where it left off
    if loaded_record == True:
        model_dict = record["models"]
    else:
        if config.same_init:
            model = init_model(config, logger)
            model_dict = {model_id: copy.deepcopy(model) for model_id in user_ids}
        else:
            model_dict = {model_id: init_model(config, logger) for model_id in user_ids}
    
    def train_client(client_id, neighbors, model_dict, record, config, dataset, verbose=False):
        acc = []
        losses = []
        params_list = []

        global_kernel = None
        global_xs = None
        global_ys = None
        local_packages = []
        local_kernels = []

        # Refers to the "global" jac for this client
        global_jac = None

        averaged_weight = average_neighbor_weights(client_id, neighbors, model_dict)
        # Load aggregated weights into neighbors to evaluate jacobians, f(x), and store for later reloading
        old_neighbor_weights = load_and_deload_neighbor_weights(neighbors, model_dict, averaged_weight)
        # Load weight into client as well
        model_dict[client_id].load_state_dict(averaged_weight)

        # Aggregate jacobians, x, y for each neighbor to simulate sending to current client
        # (Though in the implementation, we would send f(x) rather than x, we append x here for simplicity)
        # the +[client_id] is to include the client itself
        for user_id in neighbors+[client_id]:
            # Select the model with which to take jacobian
            model = model_dict[user_id]

            if verbose: logger.info("user {:d} sending jacobian".format(user_id))
            # assign_user_resource specifies some parameters for the user given their user_id
            # user_resource is a dictionary with keys: lr, device, batch_size, images, labels
            user_resource = assign_user_resource(config, user_id, 
                                    dataset["train_data"], dataset["user_with_data"])
            local_updater = LocalUpdater(config, user_resource)
            # Gets the local jacobians for a given client specified in local_updater
            local_updater.local_step(model)
            # Simulate uplink transmission
            local_package = local_updater.uplink_transmit()
            # Append this clients jacobians to the list
            local_packages.append(local_package)

            # Send local x and y
            if global_xs is None:
                global_xs = local_updater.xs
                global_ys = local_updater.ys
            else:
                global_xs = torch.vstack((global_xs, local_updater.xs))
                global_ys = torch.vstack((global_ys, local_updater.ys))            

            # del local_updater
            torch.cuda.empty_cache()
        reload_neighbor_weights(neighbors, model_dict, old_neighbor_weights)

        # Affirm that the model is the clients model
        model = model_dict[client_id]

        start_time = time.time()
        global_jac = combine_local_jacobians(local_packages)
        #del local_packages
        # Added these two lines to free up memory
        del local_package
        del local_updater
        if verbose: logger.info("compute kernel matrix")
        global_kernel = empirical_kernel(global_jac)

        if verbose: logger.info("kernel computation time {:3f}".format(time.time() - start_time))
        # Returns a function that, given t and f_0, solves for f_t
        predictor = gradient_descent_ce(global_kernel.cpu(), global_ys.cpu(), config.lr)
            
            
        # This is f^(0) (X)
        # Note: The model var still has the aggregated weight, so it can be used to evaluate the model
        # and find f0. However, in the distributed implementation, we would send the model to the client
        with torch.no_grad():
            fx_0 = model(global_xs)

        # Configure maximum t as one more than the largest tau value
        t = torch.arange(config.taus[-1]+1)

        # Create f_x using the time values and the initial f_x
        fx_train = predictor(t, fx_0.cpu())
        # fx_train = fx_train.to(fx_0)

        # Set the averaged weight as the weight to be evaluated
        init_state_dict = averaged_weight
        losses = np.zeros_like(taus, dtype=float)
        acc = np.zeros_like(taus, dtype=float)

        if verbose: logger.info("loss \tacc")
        for i, tau in enumerate(config.taus):
            # initialize the weight aggregator with current weights
            weight_aggregator = WeightMod(init_state_dict)
            global_omegas = get_omegas(t[:tau+1], config.lr, global_jac, 
                    global_ys.cpu(), fx_train[:tau+1], config.loss, 
                    model.state_dict())
            # global_omegas = get_omegas(t[:tau+1], config.lr, global_jac, 
            #         global_ys, fx_train[:tau+1], config.loss, 
            #         model.state_dict())        
            
            # Complete the sum in 9b
            weight_aggregator.add(global_omegas)
            aggregated_weight = weight_aggregator.state_dict()
            model.load_state_dict(aggregated_weight)

            output = model(global_xs)    

            loss = loss_with_output(output, global_ys, config.loss)
            # loss_fx = loss_with_output(fx_train[tau].to(global_ys), global_ys, config.loss)
            losses[i] = loss

            output = model(test_images)

            test_acc = accuracy_with_output(output, test_labels)
            acc[i] = test_acc

            if verbose: logger.info("{:.3f}\t{:.3f}".format(loss, test_acc))

            params_list.append(copy.deepcopy(aggregated_weight))

        # Get index of tau with lowest loss
        idx = np.argmin(losses)
        # Select weight parameters with lowest loss
        params = params_list[idx]

        # Select tau with lowest loss
        current_tau = taus[idx]
        current_acc = acc[idx]
        current_loss = losses[idx]

        if verbose: logger.info("current tau {:d}".format(current_tau))
        if verbose: logger.info("acc {:4f}".format(current_acc))
        if verbose: logger.info("loss {:.4f}".format(current_loss))

        # Load weight into client model
        model_dict[client_id].load_state_dict(params)

        # Return the current loss, accuracy, and tau
        # record["loss"].append(current_loss)
        # record["testing_accuracy"].append(current_acc)
        # record["taus"].append(current_tau)
        torch.cuda.empty_cache()
        return current_loss, current_acc, current_tau

    def complete_train(user_ids, comm_rounds, model_dict, record, config, dataset, verbose=False):
        # Get zeroth round loss, acc, and tau
        if record["epoch"] == 0:
            logger.info("Logging initial loss, acc, and tau")
            client_losses = []
            client_accs = []
            client_taus = []
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
                client_taus.append(-1)
                if verbose: logger.info("client {:d} loss {:.4f} acc {:.4f} tau {:d}".format(client_id, loss, acc, -1))
            
            # Get rid of unnecessary variables to free up memory
            del user_images; del user_labels; del output_on_own_data; del output_on_test_set
            # Finally, append the initial losses, accs, and taus to the record
            record["loss"].append(client_losses)
            record["testing_accuracy"].append(client_accs)
            record["taus"].append(client_taus)

        # Train on own data once before starting the rounds IF self-training is specified only for the beginning
        if config.train_on_own_data and not config.own_data_all_rounds:
            print("First round of self-training only")
            for client_id in user_ids:
                if verbose: logger.info(f"Client {client_id} training on own data")
                # [] for no neighbors
                loss, acc, tau = train_client(client_id, [], model_dict, record, config, dataset, verbose=verbose)
                if verbose: logger.info(f"Client {client_id} loss: {loss}, acc: {acc}, tau: {tau}")
            
        
        for i in range(record["epoch"],record["epoch"] + comm_rounds):
            logger.info(f"Comm Round: {i}")
            client_losses = []
            client_accs = []
            client_taus = []
            
            # If NTK self-training is specified for all rounds, train on own data at the beginning of each round
            if config.train_on_own_data and config.own_data_all_rounds and config.self_train == "ntk":
                print("Training on own data every round")
                for client_id in user_ids:
                    if verbose: logger.info(f"Client {client_id} training on own data: round {i}")
                    # [] for no neighbors
                    loss, acc, tau = train_client(client_id, [], model_dict, record, config, dataset, verbose=verbose)
                    if verbose: logger.info(f"Client {client_id} loss: {loss}, acc: {acc}, tau: {tau}")
                
            
            # Create the graph for this round
            if config.topology == "random":
                G = create_random_graph(config.users, config.p, config.graph_name)
            elif config.topology == "ring":
                G = create_ring_graph(config.users, config.graph_name)
            else: 
                raise ValueError("Invalid topology: {}".format(config.topology))
            # Train each client on neighbors + their own data
            clients_trained = 0
            for client_id in user_ids:
                neighbors = list(G.neighbors(client_id))
                if verbose: logger.info(f"Num neighbors: {len(neighbors)}")
                loss, acc, tau = train_client(client_id, neighbors, model_dict, record, config, dataset, verbose=verbose)
                client_losses.append(loss)
                client_accs.append(acc)
                client_taus.append(tau)
                if verbose: logger.info("client {:d} loss {:.4f} acc {:.4f} tau {:d}".format(client_id, loss, acc, tau))
                clients_trained += 1
                if clients_trained % 20 == 0 and verbose:
                    logger.info(f"Clients trained: {clients_trained}")
            # Create 2D array of client losses per round
            logger.info(f"Avg client loss: {np.mean(client_losses)}")
            print(f"Avg client loss: {np.mean(client_losses)}")
            logger.info(f"Avg client acc: {np.mean(client_accs)}")
            
            record["taus"].append(client_taus)

            # If self training with SGD
            if config.self_train == "sgd":            
                record["pre_sgd_loss"].append(client_losses)
                record["pre_sgd_acc"].append(client_accs)
                # Now perform self training with SGD
                if config.loss == "ce":
                    loss_fn_pytorch = nn.CrossEntropyLoss()
                else: 
                    raise NotImplementedError
                
                for client_id in user_ids:
                    self_train(model_dict[client_id], client_id, dataset, config, logger, loss_fn_pytorch, 
                               config.sgd_batch_size, config.sgd_epochs, config.sgd_lr, verbose = config.verbose)
                
                # Now, evaluate the models
                losses = []
                accs = []
                for client_id in user_ids:
                    # Get model outputs
                    output_on_test_set = model_dict[client_id](test_images)

                    # Get losses/accs, and append to list
                    loss = loss_with_output(output_on_test_set, test_labels, config.loss)
                    acc = accuracy_with_output(output_on_test_set, test_labels)
        
                    losses.append(loss)
                    accs.append(acc)

                if verbose or True: print("After SGD update: average loss: {:.4f}, Average acc: {:.4f}".format(np.mean(losses), np.mean(accs)))
                record["loss"].append(losses)
                record["testing_accuracy"].append(accs)
            
            # If not training with SGD
            else:
                record["loss"].append(client_losses)
                record["testing_accuracy"].append(client_accs)
            # Save Model Dict
            record["models"] = model_dict
            record["epoch"] += 1
            # Save the record
            save_record(config, record)

    complete_train(user_ids, config.rounds, model_dict, record, config, dataset, verbose=config.verbose)

def main(config_file):
    # load the configuration file
    config = load_config(config_file)
    
    logger = init_logger(config)
    
    # Define a model random to extract number of parameters for record
    if config.record_path is not None:
        record = load_record(config.record_path)
        logger.info("Loaded record from {}".format(config.record_path))
        loaded_record = True
    else:
        model = init_model(config, logger)
        record = init_record(config, model)
        loaded_record = False
    
    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    # Record topology
    record["topology"] = config.topology
    if config.self_train == "sgd":
        record["pre_sgd_loss"] = []
        record["pre_sgd_acc"] = []


    # All work is done in the train function
    start = time.time()
    train(config, logger, record, loaded_record=loaded_record)
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


