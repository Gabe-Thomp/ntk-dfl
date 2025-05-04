## NTK-DFL: Enhancing Decentralized Federated Learning in Heterogeneous Settings via Neural Tangent Kernel

🙌 Happy to announce that we were accepted to ICML 2025! 🙌 Check out the manuscript [here](https://arxiv.org/abs/2410.01922).

We introduce NTK-DFL, a decentralized federated learning paradigm that makes use of neural tangent kernel weight evolution.

**Abstract**: Decentralized federated learning (DFL) is a collaborative machine learning framework for training a model across participants without a central server or raw data exchange. DFL faces challenges due to statistical heterogeneity, as participants often possess different data distributions reflecting local environments and user behaviors. Recent work has shown that the neural tangent kernel (NTK) approach, when applied to federated learning in a centralized framework, can lead to improved performance. The NTK-based update mechanism is more expressive than typical gradient descent methods, enabling more efficient convergence and better handling of data heterogeneity. We propose an approach leveraging the NTK to train client models in the decentralized setting, while introducing a synergy between NTK-based evolution and model averaging. This synergy exploits inter-model variance and improves both accuracy and convergence in heterogeneous settings. Our model averaging technique significantly enhances performance, boosting accuracy by at least 10% compared to the mean local model accuracy. Empirical results demonstrate that our approach consistently achieves higher accuracy than baselines in highly heterogeneous settings, where other approaches often underperform. Additionally, it reaches target performance in 4.6 times fewer communication rounds. We validate our approach across multiple datasets, network topologies, and heterogeneity settings to ensure robustness and generalizability. 
<br />
### Prerequisites

1) Create a conda environment with Python 3.10
```bash
conda create -n ntkdfl python=3.10
conda activate ntkdfl
```
2) Install PyTorch in your environment using conda. Follow the official PyTorch documentation [here](https://pytorch.org/get-started/locally/).
3) Modify the name of the conda environment in the `environment.yml` file to match the name of your conda environment.
4) Install the required packages using the `environment.yml` file
```bash
conda env update -f environment.yml
```

<br />

### Example Usage

Run the example using IID Fashion-Mnist: 
```bash
python3 main.py config.yaml
```
This will load the config file `utils/config.yaml` and data from `data/fmnist/...`, specified in the config file under `train_data_dir` and `test_data_dir`. Training with the NTK-DFL with the specified setup will be performed. A record and train log will be stored in the specifed directory, defaulting to `results/`.

By modifying the `train_data_dir` and `test_data_dir` paths, you can change the dataset that you want to use. Non-IID partitions have been generated and stored in the `user_with_data` files. Specify the path to the desired partition in the `config.yaml` file under `user_data_dir` in order to select your non-IID partition.

For example, to use EMNIST, modifying the `config.yaml` to 
```
train_data_dir: data/emnist/digits/train.dat
test_data_dir:  data/emnist/digits/test.dat
```

To use a non-IID $\alpha=0.1$ partition for Fashion-MNIST, modify the `config.yaml` to 
```
user_data_dir: data/user_with_data/fmnist300/a0.1/user_dataidx_map_0.10_0.dat
```
<br />

