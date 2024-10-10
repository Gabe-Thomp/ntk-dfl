import sys
directory_path = "../"
if directory_path not in sys.path:
    # Add the directory to sys.path
    sys.path.append(directory_path)

import copy
import time
import numpy as np
import argparse
import yaml

from torch.utils import data

from utils.utils import *
from fedlearning.topology import *
from utils import load_config
from utils.validate import *
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas

# Overall training function. Takes the current record, config, and logger
# loaded_recrod is a boolean that specifies whether the record was loaded from a previous checkpoint
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
    
    # Function to train a client specified by client_id with neighbors specified by neighbors
    # Client is already initialized with the averaged model_dict of neighbors
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

        # averaged_weight = average_neighbor_weights(client_id, neighbors, model_dict)
        original_client_weight = copy.deepcopy(model_dict[client_id].state_dict())
        # Load aggregated weights into neighbors to evaluate jacobians, f(x), and store for later reloading
        old_neighbor_weights = load_and_deload_neighbor_weights(neighbors, model_dict, original_client_weight)
        # # Load weight into client as well
        # model_dict[client_id].load_state_dict(averaged_weight)

        # Aggregate jacobians, x, y for each neighbor to simulate sending to current client
        # the +[client_id] is to include the client itself
        lr = config.lr * np.exp(-config.lr_decay * record["epoch"])
        if config.verbose: logger.info("lr: {:.4f}".format(lr))
        
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
            
            if config.gpu_intensive or config.jac_calc_intensive:
                local_updater.local_step(model)
            else:
                local_updater.local_step(model, device="cpu")
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
        # Reload the old weights of the neighbors after obtaining the jacobins with respect to the client's w-bar
        reload_neighbor_weights(neighbors, model_dict, old_neighbor_weights)

        # Affirm that the model is the clients model
        model = model_dict[client_id]

        start_time = time.time()
        if config.gpu_intensive or config.ntk_intensive:
            global_jac = combine_local_jacobians(local_packages, device=config.device)
        else: 
            global_jac = combine_local_jacobians(local_packages, device="cpu")
            
        del local_packages
        # Added these two lines to free up memory
        del local_package
        del local_updater
        if verbose: logger.info("compute kernel matrix")
        global_kernel = empirical_kernel(global_jac)

        if verbose: logger.info("kernel computation time {:3f}".format(time.time() - start_time))
        # Returns a function that, given t and f_0, solves for f_t
        if config.gpu_intensive or config.deq_intensive:
            predictor = gradient_descent_ce(global_kernel, global_ys, lr)
            # Configure maximum t as one more than the largest tau value
            t = torch.arange(config.taus[-1]+1, device=config.device)
        else:
            predictor = gradient_descent_ce(global_kernel.to("cpu"), global_ys.to("cpu"), lr)
            # Configure maximum t as one more than the largest tau value
            t = torch.arange(config.taus[-1]+1, device="cpu")
        
        # This is f^(0) (X)
        with torch.no_grad():
            fx_0 = model(global_xs)

        # Create f_x using the time values and the initial f_x
        if config.gpu_intensive or config.deq_intensive:
            fx_train = predictor(t, fx_0)
        else:
            fx_train = predictor(t, fx_0.to("cpu"))
        # fx_train = fx_train.to(fx_0)

        # Set the client weight as the weight to be evaluated
        init_state_dict = original_client_weight
        losses = np.zeros_like(taus, dtype=float)
        acc = np.zeros_like(taus, dtype=float)

        if verbose: logger.info("loss \tacc")
        for i, tau in enumerate(config.taus):
            # initialize the weight aggregator with current weights
            weight_aggregator = WeightMod(init_state_dict)
            
            if config.gpu_intensive or config.deq_intensive:
                global_omegas = get_omegas(t[:tau+1], lr, global_jac, 
                        global_ys, fx_train[:tau+1], config.loss, 
                        model.state_dict())
            else:
                global_omegas = get_omegas(t[:tau+1], lr, global_jac, 
                            global_ys.to("cpu"), fx_train[:tau+1], config.loss, 
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

        # Load original weight into client model for sake of synchronous update
        model_dict[client_id].load_state_dict(original_client_weight)

        # Return the current loss, accuracy, and tau and the params
        # record["loss"].append(current_loss)
        # record["testing_accuracy"].append(current_acc)
        # record["taus"].append(current_tau)
        torch.cuda.empty_cache()
        return ((current_loss, current_acc, current_tau), params)

    def complete_train(user_ids, comm_rounds, model_dict, record, config, dataset, verbose=False):
        # Get zeroth round loss, acc, and tau
        if record["epoch"] == 0:
            logger.info("\nLogging initial loss and acc")
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
            
        
        for i in range(record["epoch"],record["epoch"] + comm_rounds):
            logger.info(f"Comm Round: {i}")
            client_losses = []
            client_accs = []
            client_taus = []
                
            # Create the graph for this round
            if config.topology == "random":
                G = create_random_graph(config.users, config.p, config.graph_name)
            elif config.topology == "ring":
                G = create_random_ring_graph(config.users, config.graph_name)
            elif config.topology == "regular":
                if config.p is not None:
                    raise ValueError("Regular graph requires d, not p")
                elif config.d is None:
                    raise ValueError("Regular graph requires d")
            
                G = create_regular_graph(config.users, config.d, config.graph_name)
                if config.verbose: logger.info(f"Creating regular graph with d={config.d}")
            else: 
                raise ValueError("Invalid topology: {}".format(config.topology))
            
            # First, all clients average with neighbors
            # Must find the avged weights then load weights to mimic synchronous averaging
            new_avged_weights = {}
            # Get new weights for all clients
            for client_id in user_ids:
                neighbors = list(G.neighbors(client_id))
                new_avged_weights[client_id] = average_neighbor_weights(client_id, neighbors, model_dict)
            # Load new weights for all clients
            for client_id in user_ids:
                if verbose: logger.info(f"Loading new averaged weights for client {client_id}")
                model_dict[client_id].load_state_dict(new_avged_weights[client_id])
            del new_avged_weights
            
            # Train each client on neighbors + their own data
            # Must find the evolved weights then load weights to mimic synchronous evolution
            new_evolved_weights = {}
            clients_trained = 0
            for client_id in user_ids:
                neighbors = list(G.neighbors(client_id))
                if verbose: logger.info(f"Num neighbors: {len(neighbors)}")
                (loss, acc, tau), best_params = train_client(client_id, neighbors, model_dict, record, config, dataset, verbose=verbose)
                new_evolved_weights[client_id] = best_params
                client_losses.append(loss)
                client_accs.append(acc)
                client_taus.append(tau)
                if verbose: logger.info("client {:d} loss {:.4f} acc {:.4f} tau {:d}".format(client_id, loss, acc, tau))
                clients_trained += 1
                if clients_trained % 20 == 0 and verbose:
                    logger.info(f"Clients trained: {clients_trained}")
            # Load new weights for all clients
            for client_id in user_ids:
                if verbose: logger.info(f"Loading new evolved weights for client {client_id}")
                model_dict[client_id].load_state_dict(new_evolved_weights[client_id])
            del new_evolved_weights
            
            # Get the accuracy of aggregated global model
            # Init model to load aggregated state dict
            temp_global_model = init_model(config, logger)
            temp_global_model.load_state_dict(average_neighbor_weights(0, user_ids[1:], model_dict))
            # Evaluate output
            global_output = temp_global_model(test_images)
            global_acc = accuracy_with_output(global_output, test_labels)
            

            # Create 2D array of client losses per round            
            logger.info(f"Avg client loss: {np.mean(client_losses)}")
            logger.info(f"Avg client acc: {np.mean(client_accs)}")
            logger.info(f"Aggregated model acc: {global_acc}")
            record["loss"].append(client_losses)
            record["testing_accuracy"].append(client_accs)
            record["taus"].append(client_taus)
            # Record the global accuracy
            if 'aggregated_accs' in record:
                record['aggregated_accs'].append(global_acc)
            else:
                record['aggregated_accs'] = [global_acc]
            
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
    # logger.info("Loaded configuration from {}".format(config_file))
    logger.info("Dataset path: {}".format(config.train_data_dir))
    # if config.user_with_data == "":
    #     logger.info("IID Dataset")
    # else:
    #     logger.info(f"Using \"{config.user_with_data}\" premade Non-IID dataset")

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
        torch.backends.cudnn.deterministic = True

    # Record topology
    record["topology"] = config.topology
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
    main(args.config_file)


