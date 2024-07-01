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


# Function to cluster participants in a given communication round into group
def cluster_participants(cluster_size, participating_users):
        np.random.shuffle(participating_users)
        num_participators = len(participating_users)
        num_clusters = num_participators // cluster_size
        clusters = np.array_split(participating_users, num_clusters)
        return clusters

# Function to average the weights of the models in a cluster (not a weighted average)
def average_cluster_weights(cluster_ids, model_dict):
    # Average the weights of the models in the cluster
    weight_dict = copy.deepcopy(model_dict[cluster_ids[0]].state_dict())
    weight_aggregator = WeightMod(weight_dict)
    for user_id in cluster_ids[1:]:
        weight_aggregator.add(copy.deepcopy(model_dict[user_id].state_dict()))
    weight_aggregator.mul(1.0/len(cluster_ids))
    return weight_aggregator.state_dict()

# Function to load the a weight_dict into models specified in cluster_ids
def load_cluster_weights(cluster_ids, model_dict, weight_dict):
    # Load the averaged weights to the models in the cluster
    for user_id in cluster_ids:
        model_dict[user_id].load_state_dict(weight_dict)
    return None

def get_accuracies(model_ids, model_dict, test_images, test_labels):
    '''
    Get the accuracies of the models in the model_dict specified by the model_ids
    returns a list of accuracies
    '''
    acc = []
    for user_id in model_ids:
        model = model_dict[user_id]
        output = model(test_images)
        test_acc = accuracy_with_output(output, test_labels)
        acc.append(test_acc)
    return acc



def train(config, logger, record):

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
    if config.same_init == True:
        model = init_model(config, logger)
        model_dict = {model_id: copy.deepcopy(model) for model_id in user_ids}
    else:
        model_dict = {model_id: init_model(config, logger) for model_id in user_ids}
    
    
    # Create a list to track the number of training rounds for each user
    model_training_rounds = [0]*config.users
    
    # Select (one more than) the number of users to participate in each communication round
    num_participators = int(config.part_rate*config.users) 

    def train_cluster(cluster_ids, model_dict, model_training_rounds, verbose=False):
        acc = []
        losses = []
        params_list = []

        global_kernel = None
        global_xs = None
        global_ys = None
        local_packages = []
        local_kernels = []

        global_jac = None

        # Find intra-cluster average weights
        averaged_weight = average_cluster_weights(cluster_ids, model_dict)
        # Load averaged weight into clusters
        load_cluster_weights(cluster_ids, model_dict, averaged_weight)
        for user_id in cluster_ids:
            # Select the model with which to take jacobian
            model = model_dict[user_id]
            
            if verbose: print("user {:d} updating".format(user_id))
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

        # Create NTK from cluster jacobians
        if verbose: print("compute kernel matrix")
        global_kernel = empirical_kernel(global_jac)

        if verbose: print("kernel computation time {:3f}".format(time.time() - start_time))

        # Returns a function that, given t and f_0, solves for f_t
        predictor = gradient_descent_ce(global_kernel.cpu(), global_ys.cpu(), config.lr)
        
        # This is f^(0) (X)
        # Note: If all models in the cluster have the same initial weights (after aggregated weights are loaded), 
        # they also have the same weight evolution under the NTK.
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

        if verbose: print("loss \tacc")
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

            if verbose: print("{:.3f}\t{:.3f}".format(loss, test_acc))

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
        logger.info("acc {:4f}".format(current_acc))
        if verbose: logger.info("loss {:.4f}".format(current_loss))
        
        # Load weights into models in cluster
        load_cluster_weights(cluster_ids, model_dict, params)
        
        # Increment training round for each model in cluster
        for user_id in cluster_ids:
            model_training_rounds[user_id] += 1
        
        # Return the current loss, accuracy, and tau
        record["loss"].append(current_loss)
        record["testing_accuracy"].append(current_acc)
        record["taus"].append(current_tau)

        torch.cuda.empty_cache()

    for round in range(config.rounds):
        logger.info("Comm Round {:d}".format(round))
        # Select participating users
        participating_users = np.random.choice(user_ids, num_participators-1, replace=False)
        # Cluster participating users
        clusters = cluster_participants(config.cluster_size, participating_users)
        for cluster in clusters:
            train_cluster(cluster, model_dict, model_training_rounds)
        participating_accs = get_accuracies(participating_users, model_dict, test_images, test_labels)
        record["model_accs_per_round"].append(get_accuracies(user_ids, model_dict, test_images, test_labels))
        logger.info(f"Participating Accs: {np.mean(participating_accs)}")
        logger.info("-"*80)
        
    # Save model_num_training_rounds
    record["model_num_training_rounds"] = model_training_rounds
    
    
def main(config_file):
    # load the configuration file
    config = load_config(config_file)
    
    logger = init_logger(config)
    
    # Define a model random to extract number of parameters for record
    model = init_model(config, logger)
    record = init_record(config, model)
    
    if config.device == "cuda":
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True

    # All work is done in the train function
    start = time.time()
    train(config, logger, record)
    end = time.time()
    # Save record to local directory
    save_record(config, record)
    
    # Log the time taken
    logger.info("{:.3f} mins has elapsed.".format((end-start)/60))

if __name__ == "__main__":
    main("config_cluster.yaml")