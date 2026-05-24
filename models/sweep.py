from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from models.create_datasets import trajectory_sweep_dataset
from models.trajectory_model.experiments import run_trajectory_model_experiments

BASE_DIR = Path("trajectory/runs12")
DATA_DIR = Path("dataset/trajectory2.0")
LOG_DIR = BASE_DIR / "logs"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
PLOTS_DIR = BASE_DIR / "plots"
FORWARD_MODEL_PATH = Path("models/forward.pt")
INVERSE_FORWARD_MODEL_PATH = Path("trajectory/models/inverse2.pt")

def save_progress_plots(
        results_csv: str,
        output_png: str,
):
    """
    Plotting function for visualisation of validation losses throughout the sweep.

    :param results_csv: CSV file path.
    :param output_png: Output image file path.
    """
    results_csv = Path(results_csv)
    if not results_csv.exists():
        print("No sweep CSV yet.")
        return

    df = pd.read_csv(results_csv)
    if df.empty:
        print("Sweep CSV is empty.")
        return
    values = df.columns.values.tolist()
    graphs = []
    for value in values:
        if "mae" in value or "l2" in value or "epoch" in value:
            continue
        if not "best" in value:
            continue
        graphs.append(value)

    fig = plt.figure(figsize=(10, 20))

    for idx, value in enumerate(graphs):
        plt.subplot(len(graphs), 1, idx + 1)

        plt.plot(df["run_id"], df[value], marker="")
        plt.xlabel("run_id")
        plt.ylabel(value)
        plt.grid(True)
    plt.title("Forward model sweep progress")

    plt.savefig(output_png, bbox_inches="tight")

if __name__ == "__main__":
    # Creates datasets and runs sweep defined in experiments
    train_dataset, val_dataset, _ = trajectory_sweep_dataset(
        h5_file_path=str(DATA_DIR / "worker_1.h5"),
        master_index_df_path=str(DATA_DIR / "trajectory_master_index.csv"),
        n_trajectories=13000,
        target_no_obstacle_ratio=0.1,
        seed=1,
    )

    run_trajectory_model_experiments(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        results_dir=str(RESULTS_DIR),
        checkpoint_dir=str(CHECKPOINT_DIR),
        log_dir=str(LOG_DIR),
        plots_dir=str(PLOTS_DIR),
        forward_model_path=str(FORWARD_MODEL_PATH),
        inverse_model_path=str(INVERSE_FORWARD_MODEL_PATH),
        after_run_callback=lambda csv_path: save_progress_plots(csv_path, str(PLOTS_DIR / "trajectory_model_progress.png")),
    )

    # Runs cross-validation for defined config
    # cfg = TrajectoryModelTrainConfig(
    #     num_epochs=50,
    #     learning_rate=1e-4,
    #     optimizer="AdamW",
    #     n_timesteps=50,
    #     d_gru=768,
    #     n_gru=2,
    #     hidden_head_dimension=256,
    # )
    #
    # cross_validation(
    #     cfg_idx=5,
    #     cfg=cfg,
    #     data_path=str(DATA_DIR / "worker_1.h5"),
    #     index_file_path=str(DATA_DIR / "trajectory_master_index.csv"),
    #     log_dir=str(LOG_DIR),
    #     checkpoint_dir=str(CHECKPOINT_DIR),
    #     results_dir=str(RESULTS_DIR),
    #     plots_dir=str(PLOTS_DIR),
    #     forward_model_path=str(FORWARD_MODEL_PATH),
    #     inverse_model_path=str(INVERSE_FORWARD_MODEL_PATH),
    # )
