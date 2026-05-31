# bc_thesis_source_code
Source code for bachelor's thesis. Includes data generation scripts with a custom wrapper for the simulated environment, model implementations, training and validation scripts, and tests.

## Installation

### Requirements

- Python 3.10+
- Python 3.11+ recommended

The project dependencies are listed in `requirements.txt`.

PyTorch is installed separately because the required package depends on the user's hardware and CUDA version.

### Clone repository

```bash
git clone https://github.com/krispik1/bc_thesis_source_code.git
cd bc_thesis_source_code
```

### Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install PyTorch

Install a PyTorch version compatible with your hardware.

See the official PyTorch installation guide:

https://pytorch.org/get-started/locally/

### Install modified myGym

This repository depends on a modified version of myGym included in the source tree.

The original project is available at https://github.com/incognite-lab/myGym.

```bash
python -m pip install -e ./myGym
```

### Install remaining dependencies

```bash
python -m pip install -r requirements.txt
```

## Source code structure

```text
myGym/ # modified version myGym environment
babbling_explorer/ # generator of babbling observations
  explorer.py
  poisson_sampler.py
  proxy_ensemble.py
  replay_buffer.py

dataset_writer/ # HDF5 dataset writers
  buffer.py
  dataset_schemas.py
  h5_writer.py
  writer_manager.py
  
models/ # implementation and experiments with neural architectures
  forward_model/
    config.py
    experiments.py
    forward_model.py
    forward_model_training.py
    loss_function.py
  inverse_model/
    config.py
    experiments.py
    inverse_model.py
    inverse_model_training.py
    loss_function.py
  trajectory_model/
    config.py
    experiments.py
    trajectory_model.py
    trajectory_model_training.py
    loss_function.py
  config.py
  create_datasets.py
  cross_validation.py
  datasets.py
  helpers.py
  sweep.py
 
tests/ # testing scripts for the models
  forward_model_error_propagation.py
  inference.py
  one_step_test.py
  rectification_error.py
  trajectory_geometry_test.py
  trajectory_simulator_test.py

trajectory_generator/ # generator of trajectory episodes
  policies/
    one_euro_filter.py
    rrt_policy.py
  dataset_generator.py
  episode_runner.py
  planner.py

config.py 
create_master_index_files.py
dataset_types.py
geometry_helpers.py
parent_launcher.py # launcher for multi-process generation
README.md
training_data_generation.json
worker.py
wrapper.py # wrapper for simulated environment
```

## Running dataset generation

Use:

```bash
python parent_launcher.py training_data_generation.json
```

The launcher reads the JSON config, starts workers for each configured generation mode, merges the worker files into one final HDF5 dataset per mode, and rebuilds the master episode index.

### Dataset generation configuration

The main config is `training_data_generation.json`.

Important fields:

```json
{
  "env_config": {},
  
  "obstacle_present": 0,
  
  "modes"                         : ["trajectory", "babbling"],
  "n_datapoints_per_mode"         : [12000, 500000],
  "n_episodes_per_setup_per_mode" : [1, 1],
  "n_workers_per_mode"            : [6, 1],
  
  "target_collision_ratio"    : 0.0,
  "target_n_waypoints_ratio"  : [1.0, 0.0, 0.0, 0.0],
  
  "interval_n_steps_per_exploration"  : [1, 1],
  "region_ratio"                      : [0.35, 0.45, 0.2],
  
  "path": "dataset"
}
```

```env_config``` is the configuration file for the simulated environment. ```obstacle_present``` is a flag indicating whether
obstacle is present, if not, trajectory planner produces only collision-free trajectories and explorer does not use
region categories. 

```target_collision_ratio``` is a ratio of colliding trajectories in the dataset, ```target_n_waypoints_ratio``` is the ratio of 
number of detour waypoints added during trajectory planning. 

```region_ratio``` is the ratio of region representation in
dataset - FAR, NEAR, and CONTACT. ```interval_n_steps_per_exploration``` is the interval from which an integer is randomly
chosen, representing the number of steps for the babbling episode.

Currently implemented modes are ```babbling``` and ```trajectory``` generation. One worker for the former mode is 
recommended for the current ```explorer.py``` implementation.

### Output structure of dataset generation

Each run gets its own run folder. The final combined file is kept outside the run folders and grows over multiple runs.

Example:

```text
dataset/
  trajectory/
    trajectory_dataset.h5
    trajectory_master_index.csv
    runs/
      20260531_120000/
        worker_1.h5
        worker_2.h5
      20260531_150000/
        worker_1.h5
        worker_2.h5

  babbling/
    babbling_dataset.h5
    babbling_master_index.csv
    runs/
      20260531_120000/
        worker_1.h5
        worker_2.h5
```

Meaning:

- `*_dataset.h5` is the final accumulated dataset used for training.
- `*_master_index.csv` indexes all episodes in the final accumulated dataset.
- `runs/<run_id>/worker_*.h5` are the raw worker outputs for each individual run.

### HDF5 layout

The generated HDF5 files contain table-like groups such as:

```text
env/ # descriptors of env, names and action in datatypes
    robot_name
    robot_action
    robot_init
    goal_obj_name
    goal_obj6D
    obstacle_name
    obstacle6D
episodes/ # describes block of transitions as episode + flags
    env_id
    ep_start
    ep_len
    planner
    mode
    collision
    success
transitions/ # state->action(both desired and realised)->next state
    joints_angles_t
    ee6D_t
    goal_obj6D_t
    obstacle6D_t
    mgt_t
    joints_angles_t1
    ee6D_t1
    goal_obj6D_t1
    obstacle6D_t1
    mgt_t1
    desired_delta_q
    delta_q
    delta_mgt
    collision
```

### Master index file

The master index file is created by ```build_master_episode_index``` in ```create_master_index_files.py``` using HDF5 files
following the aforementioned schemas. After every run, the master index is rebuilt from the final HDF5 file.

It stores episode-level metadata, including:

- source file path ```file_path```
- episode ID ```episode_id``` - pointer to ```episodes``` group in HDF5 file
- integer environment setup ID ```env_id``` - pointer to ```env``` group in HDF5 file
- unique string setup ID ```setup_id``` based on goal and obstacle object 6D poses
- episode start index ```ep_start``` and episode length ```ep_len```
- obstacle presence flag ```obstacle_active```
- goal and obstacle object 6D poses as JSONs ```goal_obj6D_json``` and ```obstacle6D_json```
- planner/mode/collision/success flags when available

## Tuning of the models

Currently implemented only through manual setting. ```sweep.py``` is used to run grid-search experimentation of
chose hyperparameters and also cross-validation of selected configs. Global variables set the various directories.
Inside individual model's ```experiments.py``` file, a hyperparameter grid can be defined. TODO: exeriments and
validation through input JSON configs.

### Implementation of models

Each model architecture has a directory ```*_model/``` with an implementation in ```*_model.py```. ```*_model_training.py```
contains fitting of the model with runs using training and validation epoch with the corresponding loss function in
```loss_function.py```. The training file also includes computation of evaluation metrics and additional algorithms such
as rectification process. Concrete hyperparameter configuration of the model is defined in ```config.py```.
```experiments.py``` handles loading and experiments with the model using grid search.

### Datasets

```datasets.py``` defines a new Dataset class for each type of model using HDF5 file. ```create_datasets.py``` initializes
these datasets or creates list of indices based on master index file that serve as pointers to the HDF5 files.

## Testing of the trained models

Currently, the following testing scripts are implemented.

- ```forward_model_error_propagation.py``` - test forward model error propagation divided into no obstacle and obstacle
present propagation
- ```one_step_test.py``` - test forward model/inverse model error for single transition prediction divided into obstacle-free
environment, obstacle collision-free and obstacle colliding transitions
- ```rectification_error.py``` - test rectification algorithm error for combination of forward+inverse model
- ```trajectory_geometry_test.py``` - test trajectory model's predictions in terms of end-effector position geometry 
divided into predictions in no obstacle and obstacle present environments
- ```trajectory_simulator_test.py``` - test trajectory model by executing predicted trajectories in simulated environment
with execution modes: joint-angle configuration and end-effector position + inverse
kinematics 
- ```inference.py``` - test inference time of the models divided into obstacle-free environment and with obstacle
present predictions

Tests produce ```.csv``` and ```.tex``` results table and require changing the model
path and name, set desired seeds and output paths in the global variables. TODO: allow configuration through input JSON.

## Citation

If you use this code in academic work, please cite the associated bachelor's thesis.