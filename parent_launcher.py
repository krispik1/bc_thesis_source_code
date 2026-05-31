import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

import commentjson
import multiprocessing as mp

import h5py
import numpy as np

from config import TrainingDatasetGenerationConfig
from create_master_index_files import build_master_episode_index
from worker import worker

H5_SUFFIXES = {".h5", ".hdf5"}

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

def split_datapoints_for_workers(
        total_datapoints: int,
        n_workers: int
) -> List[int]:
    """
    Split datapoints across workers without dropping the remainder.

    :param total_datapoints: Total number of datapoints.
    :param n_workers: Number of workers.
    :return: Split datapoints across workers.
    """
    base, remainder = divmod(total_datapoints, n_workers)
    return [base + (1 if i < remainder else 0) for i in range(n_workers)]

def _worker_h5_paths(
        run_mode_dir: Path
) -> List[Path]:

    return sorted(
        p for p in run_mode_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in H5_SUFFIXES
        and p.name.startswith("worker_")
    )

def _ensure_schema(
        src_group: h5py.Group,
        dst_group: h5py.Group,
        group_name: str,
        src_path: Path
) -> None:

    if set(src_group.keys()) != set(dst_group.keys()):
        raise ValueError(f"Schema mismatch in {group_name} while merging {src_path}")

    for name in src_group.keys():
        src_ds = src_group[name]
        dst_ds = dst_group[name]
        if src_ds.shape[1:] != dst_ds.shape[1:] or src_ds.dtype != dst_ds.dtype:
            raise ValueError(f"Dataset mismatch for {group_name}/{name} while merging {src_path}")

def _copy_attrs(
        src: h5py.AttributeManager,
        dst: h5py.AttributeManager
) -> None:

    for key, value in src.items():
        dst[key] = value

def _append_dataset(
        dst_ds: h5py.Dataset,
        data: np.ndarray
) -> None:

    old_size = dst_ds.shape[0]
    new_size = old_size + data.shape[0]
    dst_ds.resize((new_size,) + dst_ds.shape[1:])
    dst_ds[old_size:new_size] = data

def _create_like(
        dst_group: h5py.Group,
        name: str,
        src_ds: h5py.Dataset
) -> h5py.Dataset:

    kwargs = {
        "shape": (0,) + src_ds.shape[1:],
        "maxshape": (None,) + src_ds.shape[1:],
        "dtype": src_ds.dtype,
        "chunks": src_ds.chunks or (min(max(src_ds.shape[0], 1), 4096),) + src_ds.shape[1:],
    }
    if src_ds.compression is not None:
        kwargs["compression"] = src_ds.compression
    if src_ds.compression_opts is not None:
        kwargs["compression_opts"] = src_ds.compression_opts
    if src_ds.shuffle:
        kwargs["shuffle"] = True
    if src_ds.fletcher32:
        kwargs["fletcher32"] = True

    ds = dst_group.create_dataset(name, **kwargs)
    _copy_attrs(src_ds.attrs, ds.attrs)
    return ds

def _first_dataset_len(
        group: h5py.Group
) -> int:

    if not group.keys():
        return 0
    first = next(iter(group.keys()))
    return int(group[first].shape[0])

def _init_output_from_first_worker(
        dst: h5py.File,
        src: h5py.File
) -> None:

    _copy_attrs(src.attrs, dst.attrs)

    for group_name, src_group in src.items():
        dst_group = dst.create_group(group_name)
        _copy_attrs(src_group.attrs, dst_group.attrs)

        for ds_name, src_ds in src_group.items():
            _create_like(dst_group, ds_name, src_ds)

def append_worker_h5_files(
        worker_paths: List[Path],
        final_path: Path
)-> Path:
    """
    Combine all worker HDF5 files for one mode into one HDF5 file.

    The merge preserves the original groups/datasets, appends rows in worker order, and fixes cross-table pointers:
    - episodes/env_id is shifted by the number of env rows already written.
    - episodes/ep_start is shifted by the number of transition rows already written.
    - Existing final_path is preserved and extended.
    - Only the current run's worker files should be passed here.
    """

    worker_paths = sorted(Path(p) for p in worker_paths)
    if not worker_paths:
        raise FileNotFoundError("No worker HDF5 files found for this run.")

    final_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(final_path, "a") as dst:
        output_initialized = len(dst.keys()) > 0

        env_offset = _first_dataset_len(dst["env"]) if output_initialized and "env" in dst else 0
        transition_offset = (
            _first_dataset_len(dst["transitions"])
            if output_initialized and "transitions" in dst
            else 0
        )

        for src_path in worker_paths:
            with h5py.File(src_path, "r") as src:
                if not output_initialized:
                    _init_output_from_first_worker(dst, src)
                    output_initialized = True
                else:
                    for group_name, src_group in src.items():
                        if group_name not in dst:
                            raise ValueError(f"Missing group {group_name!r} in final file.")
                        _ensure_schema(src_group, dst[group_name], group_name, src_path)

                current_env_rows = _first_dataset_len(src["env"])
                current_transition_rows = _first_dataset_len(src["transitions"])

                for group_name, src_group in src.items():
                    for ds_name, src_ds in src_group.items():
                        data = src_ds[...]

                        if group_name == "episodes" and ds_name == "env_id":
                            data = data + env_offset
                        elif group_name == "episodes" and ds_name == "ep_start":
                            data = data + transition_offset

                        _append_dataset(dst[group_name][ds_name], data)

                env_offset += current_env_rows
                transition_offset += current_transition_rows

    return final_path

def generate_datasets(
        config_path: str,
):
    """
    Parent launcher that creates context for multiple processes. These multiple processes are the workers that
    create datasets. Each worker has its own myGym environment and calculates trajectories which will be run in
    episodes. These episodes are then stored in HDF5 files. To avoid concurrency, each worker also has its own
    HDF5 file and writer.
    """

    # Number of workers based on CPU cores
    cfg_dict = load_config(config_path)
    cfg = TrainingDatasetGenerationConfig(**cfg_dict)

    # If environment should be obstacle-free, remove obstacle objects
    if not cfg.obstacle_present:
        cfg.env_config["distractors"]["list"] = []

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run different modes of data generation
    modes = cfg.modes
    base_path = Path(cfg.path)
    outputs: Dict[str, Dict[str, str]] = {}

    n_workers_per_mode = cfg.n_workers_per_mode
    n_datapoints_per_mode = cfg.n_datapoints_per_mode
    n_episodes_per_setup_per_mode = cfg.n_episodes_per_setup_per_mode

    for mode, n_workers, n_datapoints, n_episodes_per_setup in zip(
            modes,
            n_workers_per_mode,
            n_datapoints_per_mode,
            n_episodes_per_setup_per_mode
    ):
        n_datapoints_per_worker = split_datapoints_for_workers(n_datapoints, n_workers)

        mode_dir = base_path / mode
        final_h5_path = mode_dir / f"{mode}_dataset.h5"
        master_index_path = mode_dir / f"{mode}_master_index.csv"

        run_base_path = mode_dir / "runs" / run_id
        run_mode_dir = run_base_path / mode
        run_mode_dir.mkdir(parents=True, exist_ok=True)

        # If one worker needed, don't multiprocess
        if n_workers == 1:
            worker(1, cfg, mode, str(run_mode_dir), n_datapoints_per_worker[0], n_episodes_per_setup)

        # Multiprocess workers, each with its own environment and generators
        else:
            ctx = mp.get_context("spawn")
            processes = []
            for i in range(n_workers):
                process = ctx.Process(
                    target=worker,
                    args=(i + 1, cfg, mode, str(run_mode_dir), n_datapoints_per_worker[i], n_episodes_per_setup)
                )
                process.start()
                processes.append(process)

            for process in processes:
                process.join()
                if process.exitcode != 0:
                    raise Exception(f"Process exited with code {process.exitcode}")

        worker_paths = _worker_h5_paths(run_mode_dir)

        append_worker_h5_files(worker_paths, final_h5_path)
        build_master_episode_index([final_h5_path], master_index_path)

        outputs[mode] = {
            "run_id": run_id,
            "run_worker_dir": str(run_mode_dir),
            "combined_h5": str(final_h5_path),
            "master_index": str(master_index_path),
        }

    return outputs

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Parent launcher that creates context for multiple processes.")
        parser.add_argument("config_path", type=str, help="Path to config file")

        args = parser.parse_args()
        generated = generate_datasets(args.config_path)

        for mode, paths in generated.items():
            print(
                f"{mode}: run_id={paths['run_id']} "
                f"workers={paths['run_worker_dir']} "
                f"combined_h5={paths['combined_h5']} "
                f"master_index={paths['master_index']}"
            )

    except KeyboardInterrupt:
        print("Interrupted.")