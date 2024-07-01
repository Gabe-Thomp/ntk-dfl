import copy
import time
import time
import numpy as np
import argparse

from torch.utils import data

from utils.utils import *
from utils import load_config
from utils.validate import *
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from random_graph import average_neighbor_weights


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

def self_train(user_model, user_id, dataset, config, logger, loss_fn, batch_size=32, epochs=1, lr = 0.001): 
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

            if batch_idx % 100 == 0 and config.verbose:
                print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(user_data_loader.dataset)} ({100. * batch_idx / len(user_data_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    if config.verbose: print()


def train(model, config, logger, record):
    # initialize user_ids
    user_ids = np.arange(0, config.users)
    num_participators = int(config.part_rate*config.users) 

    # load the dataset
    # dataset object is a dictionary with keys: train_data, test_data, user_with_data
    # user_with_data is a dictionary with keys: userID:sampleID
    # For example, in the IID setting ID's are just assigned like 0, 1, 2, 3, ...
    dataset = assign_user_data(config, logger)
    test_images = torch.from_numpy(dataset["test_data"]["images"]).to(config.device)
    test_labels = torch.from_numpy(dataset["test_data"]["labels"]).to(config.device)

    # tau candidates 
    taus = np.array(config.taus)

    # before optimization, report the result first
        

    # start communication training rounds
    for comm_round in range(config.rounds):

        logger.info("Comm Round {:d}".format(comm_round))
        # Sample random subset of users
        np.random.shuffle(user_ids)
        participator_ids = user_ids[:num_participators]

        # schedule some values to pick up
        acc = []
        losses = []
        params_list = []

        global_kernel = None
        global_xs = None
        global_ys = None
        local_packages = []
        local_kernels = []

        # Added for memory management
        global_jac = None
        for user_id in participator_ids:
            # print("user {:d} updating".format(user_id))
            
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

        start_time = time.time()
        global_jac = combine_local_jacobians(local_packages)

        
        del local_packages
        # Added these two lines to free up memory
        del local_package
        del local_updater


        print("compute kernel matrix")
        global_kernel = empirical_kernel(global_jac)

        print("kernel computation time {:3f}".format(time.time() - start_time))

        # Returns a function that, given t and f_0, solves for f_t
        predictor = gradient_descent_ce(global_kernel.cpu(), global_ys.cpu(), config.lr)
        
        # This is f^(0) (X)
        with torch.no_grad():
            fx_0 = model(global_xs)

        # Configure maximum t as one more than the largest tau value
        t = torch.arange(config.taus[-1]+1)
        
        # Create f_x using the time values and the initial f_x
        fx_train = predictor(t, fx_0.cpu())
        # fx_train = fx_train.to(fx_0)
        
        # Use current weights to pass to the optimizer
        init_state_dict = copy.deepcopy(model.state_dict())

        losses = np.zeros_like(taus, dtype=float)
        acc = np.zeros_like(taus, dtype=float)

        print("loss \tacc")

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

            print("{:.3f}\t{:.3f}".format(loss, test_acc))

            params_list.append(copy.deepcopy(aggregated_weight))
        
        # Get index of tau with lowest loss
        idx = np.argmin(losses)
        # Select weight parameters with lowest loss
        params = params_list[idx]

        # Select tau with lowest loss
        current_tau = taus[idx]
        current_acc = acc[idx]
        current_loss = losses[idx]


        logger.info("current tau {:d}".format(current_tau))
        logger.info("acc {:4f}".format(current_acc))
        logger.info("loss {:.4f}".format(current_loss))
        
        record["pre_sgd_loss"].append(current_loss)
        record["pre_sgd_acc"].append(current_acc)
        
        # Load weights into model
        model.load_state_dict(params)
        torch.cuda.empty_cache()

        # Now, perform SGD locally
        # Create temporary client models
        temp_models = {}
        for id in participator_ids:
            temp_models[id] = copy.deepcopy(model)
        
        if config.loss == "ce":
            loss_fn_pytorch = nn.CrossEntropyLoss() 
        else: 
            raise NotImplementedError
        
        for user_id in participator_ids:
            self_train(user_model=temp_models[user_id], user_id=user_id, 
                       dataset=dataset, config=config, logger=logger, 
                    loss_fn=loss_fn_pytorch, batch_size=config.local_batch_size, 
                    epochs=config.local_epochs, lr=config.local_lr)
        
        # Get losses and accuracies
        losses = []
        accs = []
        for user_id in participator_ids:
            # Get model outputs
            output_on_test_set = temp_models[user_id](test_images)

            # Get losses/accs, and append to list
            loss = loss_with_output(output_on_test_set, test_labels, config.loss)
            acc = accuracy_with_output(output_on_test_set, test_labels)
            losses.append(loss)
            accs.append(acc)

        print("Average loss over users: {:.4f}, Average acc over users: {:.4f}".format(np.mean(losses), np.mean(accs)))
        
        # Server would average all weights
        avged_dict = average_neighbor_weights(participator_ids[0], participator_ids[1:], temp_models)
        avged_dict_model = init_model(config, logger)
        avged_dict_model.load_state_dict(avged_dict)
        # Get model outputs
        output_on_test_set = avged_dict_model(test_images)

        # Get losses/accs, and append to list
        loss = loss_with_output(output_on_test_set, test_labels, config.loss)
        acc = accuracy_with_output(output_on_test_set, test_labels)
        print("After averaging client weights, loss: {:.4f}, acc: {:.4f}".format(loss, acc))
        
        # Load weights into model
        model.load_state_dict(avged_dict)    
        # del params_list

        record["loss"].append(loss)
        record["testing_accuracy"].append(acc)
        record["taus"].append(current_tau)

        logger.info("-"*80)
        
        # Get rid of this big dictionary of models
        del temp_models

def main(config_file):
    # load the configuration file
    config = load_config(config_file)
    
    logger = init_logger(config)
    
    model = init_model(config, logger)

    record = init_record(config, model)
    # Add dict to record for pre-SGD losses and accuracies
    record["pre_sgd_loss"] = []
    record["pre_sgd_acc"] = []


    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    # All work is done in the train function
    start = time.time()
    train(model, config, logger, record)
    end = time.time()
    save_record(config, record)
    logger.info("{:.3f} mins has elapsed.".format((end-start)/60))

if __name__ == "__main__":
    print("Activating main_self-train.py")
    parser = argparse.ArgumentParser(description="Parse particular config file for federated learning trial")
    parser.add_argument('config_file', type=str, help="The path to the configuration file.")
    args = parser.parse_args()
    
    print("Config file: ", args.config_file)
    main(args.config_file)