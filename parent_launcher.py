from typing import Dict, Any

import commentjson
import multiprocessing as mp

from config import TrainingDatasetGenerationConfig
from worker import worker

def load_config(
        path: str
) -> Dict[str, Any]:
    """
    Load config from json file.

    :param path: File path.
    :return: Dict with config.
    """
    with open(path, "r") as f:
        cfg = commentjson.load(f)
    return cfg

def main():
    """
    Parent launcher that creates context for multiple processes. These multiple processes are the workers that
    create datasets. Each worker has its own myGym environment and calculates trajectories which will be run in
    episodes. These episodes are then stored in HDF5 files. To avoid concurrency, each worker also has its own
    HDF5 file and writer.
    """

    # Number of workers based on CPU cores
    cfg_dict = load_config("training_data_generation.json")
    cfg = TrainingDatasetGenerationConfig(**cfg_dict)
    n_workers = cfg.n_workers
    n_trajectories_per_worker = cfg.n_trajectories // n_workers
    n_babbles_per_worker = cfg.n_observations // n_workers
    c = 0

    # If one worker needed, don't multiprocess
    if n_workers == 1:
        worker_id = 1
        worker(worker_id, cfg, "trajectory", n_trajectories_per_worker, n_babbles_per_worker)
    # Multiprocess workers, each with its own environment and generators
    else:
        ctx = mp.get_context("spawn")
        processes = []
        for i in range(n_workers):
            process = ctx.Process(target=worker, args=(i + 6*c + 1, cfg, "trajectory", n_trajectories_per_worker, n_babbles_per_worker))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise Exception(f"Process exited with code {process.exitcode}")

if __name__ == "__main__":
    main()
