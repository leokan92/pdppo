# Post-Decision Proximal Policy Optimization with Dual Critic Networks for Accelerated Learning

## Paper Reference

To reference the paper associated with this work, please use the following citation:

```scss
To be included
```

The paper is available at [Arxiv](https://huggingface.co/papers/2504.05150)

This repository contains code and resources for research on using reinforcement learning, particularly the Post-Decision Proximal Policy Optimization (PDPPO), for the Stochastic Discrete Lot-Sizing Problem and a Frozen-Lake game.

## Project Structure
This repository consists of two main directories: `Lot-sizing` and `Lake application`, each containing the related files and folders.

### Lot-Sizing

`Lot-sizing` directory holds the following subdirectories:

- **agents**: Holds various versions of PDPPO agent implementations and utility functions.
- **cfg_env**: Includes environment settings and configurations files for the project in JSON format. Additionally, `generate_setting.py` is used for generating new environment settings.
- **cfg_sol**: Stores the solution settings in `sol_setting.json`.
- **envs**: Contains different environment definitions for the problem, like `simplePlant.py` and `singleSequenceDependentMachinePlant.py`.
- **logs**: Keeps the log files for the model training and evaluation.
- **models**: Stores various optimization models.
- **results**: After executing the experiments, the results are saved in this directory.
- **scenarioManager**: Manages different scenario setups.
- **test_functions**: Stores functions to validate the models and generate plots and tables.

### Lake Application

`Lake application` directory holds the following subdirectories:

- **agents**: Contains various versions of PDPPO agent implementations for the Lake problem.
- **envs**: Contains environment definitions, like `frozen_lake.py`.
- **logs**: Contains the log files and results from the model training and evaluation for different scenarios.
- **results**: Stores the output from experiments and relevant figures.

Root level scripts `experiments.py`, `generate_tables.py` and `plot_figure.py` are used for running experiments, generating output tables and plotting results respectively.

## Repository structure:

The main components of the repository are as follows:

```graphql
├───Lake application
│   ├───agents        # contains the implementations of various agents
│   ├───envs          # contains the FrozenLake environment implementation
│   ├───logs          # contains the logs of the agent's performance
│   └───results       # contains the results of the agent's performance
│       └───frozen_lake_PPO
└───Lot-sizing
    ├───agents        # contains the implementations of various agents
    │   └───utils     # utility functions for the agents
    ├───cfg_env       # contains the settings for the Lot-sizing environment
    │   └───setting file
    ├───cfg_sol
    ├───envs          # contains the Lot-sizing environment implementation
    ├───logs          # contains the logs of the agent's performance
    ├───models        # contains the models for the optimization problems
    ├───results       # contains the results of the agent's performance
    ├───scenarioManager # manages different scenarios for the Lot-sizing environment
    └───test_functions # contains test functions for the Lot-sizing environment
```

## Requirements:

This project uses the following main dependencies:

- [Python 3.8](https://www.python.org/downloads/)
- [numpy](https://numpy.org/)
- [gym](https://gym.openai.com/)
- [matplotlib](https://matplotlib.org/)
- [torch](https://pytorch.org/)
- gurobipy (not included in `requirements.txt` due to separate licensing)

## How to Reproduce

1. Clone the repository:

```
git clone https://github.com/username/repository.git
```

2. Navigate into the project directory:

```
cd repository
```

3. Install the required Python packages. This project was developed with Python 3.8. Substitute requirements.txt with your actual requirements file:

```
pip install -r requirements.txt
```

NOTE: You might need to replace the frozen environment file in your environment path with the frozen_lake.py provided in this repository for the Lake application to work properly.

4. Run the experiments:

```
python ./code/Lot-sizing/experiments.py
python ./code/Lake application/experiments.py
```

5. Generate the tables:

```
python ./code/Lot-sizing/generate_tables.py
python ./code/Lake application/generate_tables.py
```

6. Plot the figures:

```
python ./code/Lot-sizing/plot_figure.py
python ./code/Lake application/plot_figure.py
```

You can find the results of the experiments in the results directories in both Lot-sizing and Lake application directories.

## Reproducing Results
To reproduce the results in the logs and results folders, you would need to run the experiments with the same hyperparameters and seeds.

Please note that due to the stochastic nature of the environments and training process, the results might not be identical, but they should be within a similar range.


## Contact
For any additional questions, please open an issue in the repository or contact felizardo@ita.br

## References

and modified the PPO implementation in PyTorch from:

[Lot-sizing Environment](https://github.com/EdoF90/discrete_lot_sizing)

[PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)
