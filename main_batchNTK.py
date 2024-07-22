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

from random_graph import average_neighbor_weights
from main_self_train import NumpyDataset, numpy_to_tensor_transform, self_train

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
        
        # Configure lr with exponential decay
        if config.lr_end is not None:
            a = -config.rounds/np.log(config.lr_end/config.lr)
            lr = config.lr * np.exp(-comm_round / a)
        else: 
            lr = config.lr
        logger.info(f"Learning rate: {lr}")

        logger.info("Comm Round {:d}".format(comm_round))
        # Sample random subset of users
        np.random.shuffle(user_ids)
        participator_ids = user_ids[:num_participators]

        if config.self_train == "sgd":
            # Performance before
            output = model(test_images)
            # Get losses/accs, and append to list
            loss = loss_with_output(output, test_labels, config.loss)
            acc = accuracy_with_output(output, test_labels)
            logger.info("Before SGD, loss: {:.4f}, acc: {:.4f}".format(loss, acc))
            
            # Make a temporary client model for each user
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
                        loss_fn=loss_fn_pytorch, batch_size=config.sgd_batch_size, 
                        epochs=config.sgd_epochs, lr=lr * 0.1)
            # Combine the models
            avged_dict = average_neighbor_weights(participator_ids[0], participator_ids[1:], temp_models)
            model.load_state_dict(avged_dict)
            # Test acc
            output = model(test_images)
            # Get losses/accs, and append to list
            loss = loss_with_output(output, test_labels, config.loss)
            acc = accuracy_with_output(output, test_labels)
            logger.info("After SGD, loss: {:.4f}, acc: {:.4f}".format(loss, acc))
            del temp_models
            del avged_dict
            del output

        # Get the current model (with model parameters). This is what clients will take the jacobian with respect to.
        model_for_round = copy.deepcopy(model)
        
        for batch_round in range(config.batch_m):
            # schedule some quantities to pick up
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
                user_resource["images"] = user_resource["images"][batch_round * config.local_batch_size//config.batch_m: (batch_round+1) * config.local_batch_size//config.batch_m]
                user_resource["labels"] = user_resource["labels"][batch_round * config.local_batch_size//config.batch_m: (batch_round+1) * config.local_batch_size//config.batch_m]
                user_resource["batch_size"] = config.local_batch_size//config.batch_m
                local_updater = LocalUpdater(config, user_resource)
                
                # Gets the local jacobians for a given client specified in local_updater
                
                
                if config.gpu_intensive:
                    local_updater.local_step(model_for_round, device=config.device)
                else:
                    local_updater.local_step(model_for_round, device="cpu")
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
            if config.gpu_intensive:
                global_jac = combine_local_jacobians(local_packages, device=config.device)
            else:
                global_jac = combine_local_jacobians(local_packages, device="cpu")

            
            del local_packages
            # Added these two lines to free up memory
            del local_package
            del local_updater


            print("compute kernel matrix")
            global_kernel = empirical_kernel(global_jac)

            print("kernel computation time {:3f}".format(time.time() - start_time))

            # Returns a function that, given t and f_0, solves for f_t
            if config.gpu_intensive:
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
            
            if config.gpu_intensive:
                fx_train = predictor(t, fx_0)
            else:
                fx_train = predictor(t, fx_0.to("cpu"))
            # fx_train = fx_train.to(fx_0)
            
            # Use current weights to pass to the optimizer
            init_state_dict = copy.deepcopy(model.state_dict())

            losses = np.zeros_like(taus, dtype=float)
            acc = np.zeros_like(taus, dtype=float)

            print("loss \tacc")

            for i, tau in enumerate(config.taus):
                # initialize the weight aggregator with current weights
                weight_aggregator = WeightMod(init_state_dict)
                
                if config.gpu_intensive:
                    global_omegas = get_omegas(t[:tau+1], lr, global_jac, 
                        global_ys, fx_train[:tau+1], config.loss, 
                        model.state_dict())        
                else:
                    global_omegas = get_omegas(t[:tau+1], lr, global_jac, 
                            global_ys.to("cpu"), fx_train[:tau+1], config.loss, 
                            model.state_dict())
                    
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
            # Load weights into model
            model.load_state_dict(params)

        # del params_list

        record["loss"].append(current_loss)
        record["testing_accuracy"].append(current_acc)
        record["taus"].append(current_tau)

        logger.info("-"*80)
        torch.cuda.empty_cache()

def main(config_file):
    # load the configuration file
    config = load_config(config_file)
    
    logger = init_logger(config)
    
    model = init_model(config, logger)

    record = init_record(config, model)

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
    main("config.yaml")