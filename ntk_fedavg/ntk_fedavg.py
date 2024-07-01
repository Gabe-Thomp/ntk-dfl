import copy
import time
import time
import numpy as np
import argparse
import copy

from torch.utils import data

from utils.utils import *
from utils import load_config
from utils.validate import *
from fedlearning.model import *
from fedlearning.dataset import *
from fedlearning.evolve import *
from fedlearning.optimizer import GlobalUpdater, LocalUpdater, get_omegas


config_file = "config_fedavg_NTK.yaml"
config = load_config(config_file)

logger = init_logger(config)

model = init_model(config, logger)

record = init_record(config, model)

communication_rounds = config.rounds

if config.device == "cuda":
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

# Make a deep copy of the model
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
# ALL with the same weight initialization
model_dict = {model_id: copy.deepcopy(model) for model_id in user_ids}

# Select user_id model
def train_on_user_data(user_id, model_dict):
    model = model_dict[user_id]

    #validate_and_log(model_dict[user_id], dataset, config, record, logger)
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

    user_resource = assign_user_resource(config, user_id, 
                        dataset["train_data"], dataset["user_with_data"])
    local_updater = LocalUpdater(config, user_resource)
    # Gets the local jacobians for a given client specified in local_updater
    local_updater.local_step(model)
    # Simulate uplink transmission
    local_package = local_updater.uplink_transmit()
    local_packages.append(local_package)
    # Get local xs and ys to perform weight update
    global_xs = local_updater.xs   
    global_ys = local_updater.ys

    global_jac = combine_local_jacobians(local_packages)

    global_kernel = empirical_kernel(global_jac)

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

    #print("loss \tacc")

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

        #print("{:.3f}\t{:.3f}".format(loss, test_acc))

        params_list.append(copy.deepcopy(aggregated_weight))

    # Get index of tau with lowest loss
    idx = np.argmin(losses)
    # Select weight parameters with lowest loss
    params = params_list[idx]

    # Select tau with lowest loss
    current_tau = taus[idx]
    current_acc = acc[idx]
    current_loss = losses[idx]

    return params

for round in range(communication_rounds):
    # Stores all new weights in an array to add together
    all_updated_weights = []
    num_participators = int(config.part_rate*config.users) 
    participating_users = np.random.choice(user_ids, num_participators, replace=False)

    print(f'Round {round} with {num_participators-1} participators.')
    
    # Train all participating users
    for user_id in participating_users:
        all_updated_weights.append(train_on_user_data(user_id, model_dict))

    # Print mean test acc
    individual_test_accs = [accuracy_with_output(model_dict[user_id](test_images), test_labels) for user_id in participating_users]
    print("Mean test acc: ", np.mean(individual_test_accs))

    # Aggregate weights
    
    # Get number of data points for user 0
    n_k = float(dataset["user_with_data"][0].size)
    # get total number of data points for the round
    n = float(np.sum([dataset["user_with_data"][user_id].size for user_id in participating_users]))
    # Perform first weighted sum
    weight_aggregator = WeightMod(all_updated_weights[0])
    weight_aggregator.mul(n_k)
    # Add all weights together in weighted sum
    for i in range(1, len(all_updated_weights)):
        wi = WeightMod(all_updated_weights[i])
        n_k = float(dataset["user_with_data"][i].size)
        # Weight the weight by # of datapoints
        wi.mul(n_k)
        weight_aggregator.add(wi)

    # Scale by number of data points
    weight_aggregator.mul(1.0/n)

    # Load weights into new model to test performance
    new_agg_weights = weight_aggregator.state_dict()
    model = init_model(config, logger)
    model.load_state_dict(new_agg_weights)
    output = model(test_images)
    final_acc = accuracy_with_output(output, test_labels)
    print(f"Final acc after aggregation, round {round}: ", final_acc,"\n")

    # Load weights into all models over other clients
    for user_id in user_ids:
        model_dict[user_id].load_state_dict(new_agg_weights)
