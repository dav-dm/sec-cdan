# Sec-CDAN

## Installation

1. Place the dataset according to the path defined in [`config.yaml`](config.yaml) under `base_data_path`.  
   Each dataset must then be placed in its own folder as defined in [`dataset_config.py`](src/data/dataset_config.py).

2. It is recommended to use `virtualenv` to create an isolated Python environment:
    ```bash
    virtualenv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    It is also recommended to install [GNU parallel](https://www.gnu.org/software/parallel/).

## How To Use It

Navigate to the `src` directory and execute the main script:
```bash
cd src
python main.py
```
You can then append any of the following options to your command.  
Unless otherwise specified below, the default values are taken from [`config.yaml`](config.yaml).
The parsing logic is defined in [`args_parser.py`](src/util/args_parser.py).

### General Arguments

- `--seed [int]`: Seed for reproducibility.  
- `--gpu`: Use GPU if available.  
- `--n-thr [int]`: Number of threads.  
- `--log-dir [str]`: Log directory path.  
- `--n-tasks [1 or 2]`:  
   `1`: The model is trained on both source and target datasets at the same time.  
   `2`: The model is first trained on the source dataset, then on the target dataset.  
- `--network [str]`: Network to use. The value must match the `.py` filename in [`src/network/`](src/network/) that implements the network (e.g., `2dcnn`). 
- `--ckpt-path [str]`: Path to the `.pt` file containing the state of an approach.  
- `--skip-t1`: Skip the first task on the source dataset (used only when `--n-tasks 2`).  
- `--skip-t2`: Skip the second task on the target dataset (used only when `--n-tasks 2`).  

The [`config.yaml`](config.yaml) file contains additional parameters (e.g., for early stopping, checkpointing, etc.).

### Data-Related Arguments

In addition to the standard dataset selection arguments, the following data-related parameters can be used for data loading and processing.

- `--src-dataset [str]`: Source dataset to use.  
- `--trg-dataset [str]`: Target dataset to use.  
- `--is-flat`: Flatten the PSQ input (used for ML approaches).  
- `--num-pkts [int]`: Number of packets to consider in each biflow.  
- `--fields [FIELD] ...`: Fields used among `['PL', 'IAT', 'DIR', 'WIN', 'FLG', 'TTL']`.  
  You can specify multiple fields (e.g., `--fields PL IAT`).
- `--return-quintuple`: Return the quintuple along with the data and labels. It is mostly used for explainability purposes.

The following options are defined in [`data_module.py`](src/data/data_module.py):

- `--batch-size [int]`: Batch size for training.  
- `--adapt-batch-size [int]`: Batch size for adaptation.  
- `--num-workers [int]`: Number of worker threads for data loading.  
- `--pin-memory`: Enable pinned memory for faster data transfer to GPU.

### Approach Arguments

The two main modules responsible for the training and validation logic of an approach are:

-   `MLModule` in [`ml_module.py`](src/approach/ml_module.py) for ML approaches.
-   `DLModule` in [`dl_module.py`](src/approach/dl_module.py) for DL approaches.

To execute a specific approach located in [`src/approach/`](src/approach/), set the `--approach` argument to the corresponding `.py` file name.

The framework implements approaches from the Unsupervised Domain Adaptation literature, as well as two ML semi-supervised baselines. Each approach defines its own set of arguments, declared within its respective class. These approach-specific arguments are listed below:


#### Semi-supervised ML Baselines

1.  ***Label Propagation (LP)*** - [`label_propagation.py`](src/approach/label_propagation.py)

    -   `--lp-kernel [str]`: Kernel specification (`knn` or `rbf`).
    -   `--lp-gamma [float]`: Gamma parameter for the RBF kernel.
    -   `--lp-n-neighbors [int]`: Number of neighbors for the k-NN kernel.
    -   `--lp-max-iter [int]`: Maximum number of iterations.
    -   `--lp-tol [float]`: Convergence tolerance.

2.  ***Label Spreading (LS)*** - [`label_spreading.py`](src/approach/label_spreading.py)

    -   `--ls-kernel [str]`: Kernel specification (`knn` or `rbf`).
    -   `--ls-gamma [float]`: Gamma parameter for the RBF kernel.
    -   `--ls-n-neighbors [int]`: Number of neighbors for the k-NN kernel.
    -   `--ls-alpha [float]`: Clamping factor.
    -   `--ls-max-iter [int]`: Maximum number of iterations.
    -   `--ls-tol [float]`: Convergence tolerance.


#### Supervised DL Approaches

1.  ***Baseline (Fine-tuning and Freezing)*** – [`baseline.py`](src/approach/baseline.py)

    -   `--adaptation-strat [str]`: Strategy for adapting the model (`finetuning` or `freezing`).
    -   `--adapt-lr [float]`: Learning rate for adaptation.
    -   `--adapt-epochs [int]`: Number of epochs for adaptation.


#### Unsupervised Domain Adaptation Approaches

1.  ***Adversarial Discriminative Domain Adaptation (ADDA)*** - [`adda.py`](src/approach/adda.py)

    -   `--discr-hidden-size [int]`: Number of neurons in the hidden layer of the domain discriminator.
    -   `--adapt-lr [float]`: Learning rate for adaptation.
    -   `--adapt-epochs [int]`: Number of epochs for adaptation.
    -   `--iter-per-epoch [int]`: Number of iterations per epoch during the adaptation phase.

2.  ***Minimum Class Confusion (MCC)*** - [`mcc.py`](src/approach/mcc.py)
    -   `--mcc-t [float]`: Temperature parameter for the MCC loss.
    -   `--mcc-alpha [float]`: Weighting parameter for the adaptation loss contribution.
    -   `--adapt-lr [float]`: Learning rate for adaptation.
    -   `--adapt-epochs [int]`: Number of epochs for adaptation.
    -   `--iter-per-epoch [int]`: Number of iterations per epoch during the adaptation phase.

3.  ***Sec-Conditional Adversarial Domain Adaptation (Sec-CDAN)*** - [`sec_cdan.py`](src/approach/sec_cdan.py)
    -   `--discr-hidden-size [int]`: Number of neurons in the hidden layer of the domain discriminator.
    -   `--cdan-entropy`: If present, applies entropy conditioning.
    -   `--cdan-alpha [float]`: Weighting parameter for the adaptation loss contribution.
    -   `--adapt-lr [float]`: Learning rate for adaptation.
    -   `--adapt-epochs [int]`: Number of epochs for adaptation.
    -   `--iter-per-epoch [int]`: Number of iterations per epoch during the adaptation phase.


## Project Structure

```plaintext
sec-cdan/
├── config.yaml
├── requirements.txt
└── src/
    ├── main.py
    ├── run_experiments.sh
    ├── approach/
    ├── callback/
    ├── data/
    ├── module/
    ├── network/
    ├── trainer/
    ├── util/
```

This project is organized into multiple directories, each serving a specific purpose.

- **Approach:** located in [`src/approach/`](src/approach/), this directory contains implementations of different approaches. Each approach defines its own training, validation, and inference logic, and can be selected via the `--approach` argument.

- **Callback:** located in [`src/callback/`](src/callback/), this directory includes callback functions that are executed at specific points during the code's execution. These callbacks handle tasks such as early stopping, model checkpointing, logging outputs, and more.

- **Data:** located in [`src/data/`](src/data/), this directory is responsible for dataset management, including loading, preprocessing, and configuration. It provides utilities to read datasets, set up batch sizes, and define dataset-related parameters.

- **Module:** located in [`src/module/`](src/module/), this directory contains core components related to DL-based approaches. It includes implementations for custom loss functions, neural network head, and more.

- **Network:** located in [`src/network/`](src/network/), this directory defines different neural network architectures used in the project. It provides a selection of predefined networks and a factory method for dynamically choosing a network based on configuration.

- **Trainer:** located in [`src/trainer/`](src/trainer/), this directory contains the main training pipeline. It manages the optimization process, evaluation, and model adaptation flows.

- **Util**: located in [`src/util/`](src/util/), this directory includes utility functions that support the overall framework. It handles configuration management, argument parsing, logging, directory creation, and setting random seeds for reproducibility.

## Execution of Experiments

Experiments can be executed in two ways:

### 1. Direct Execution via `main.py`
You can manually run experiments by navigating to the `src` directory and executing:
```bash
python main.py --src-dataset <source> --trg-dataset <target> --approach <approach> --seed <seed> [other options]
```
This allows full control over individual experiment parameters.

### 2. Batch Execution via `run_experiments.sh`

For running multiple experiments in a combinatorial manner, use the [`run_experiments.sh`](src/run_experiments.sh) script:
You can manually run experiments by navigating to the `src` directory and executing:
```bash
./run_experiments.sh --src-dataset sd1,sd2 --trg-dataset td1,td2 \
    --seed 0-10 --approach adda,mcc --cpu 4 --log-keyword test
```
This script automatically generates experiment combinations based on provided parameters and runs them in parallel if [GNU parallel](https://www.gnu.org/software/parallel/) is available.
Otherwise, it falls back to `xargs`.


## Acknowledgement

We thank the following open-source implementations that were used in this work:

- [Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library)
- [GNU parallel](https://www.gnu.org/software/parallel/)

## Citation
If you use this framework, please cite:
```
@article{xx,
  title = 
}
```
