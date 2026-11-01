"""
Storage path management for experiments
"""
from pathlib import Path
import shutil

# Base data directory
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
import logging as _log; _log.getLogger(__name__).warning(f"storage.py DATA_DIR={DATA_DIR}")


def get_experiment_dir(experiment_id: str) -> Path:
    """Get the directory for an experiment."""
    return DATA_DIR / "experiments" / experiment_id


def get_lane_dir(experiment_id: str, lane_id: str) -> Path:
    """Get the directory for a lane."""
    return get_experiment_dir(experiment_id) / "lanes" / lane_id


def get_weights_dir(experiment_id: str, lane_id: str) -> Path:
    """Get the weights directory for a lane."""
    return get_lane_dir(experiment_id, lane_id) / "weights"


def get_crossarch_dir(experiment_id: str) -> Path:
    """Get the cross-architecture analysis directory for an experiment."""
    return get_experiment_dir(experiment_id) / "crossarch"


def get_initial_weights_path(experiment_id: str, lane_id: str) -> Path:
    """Get the path to the initial weights file."""
    return get_weights_dir(experiment_id, lane_id) / "initial.pt"


def get_snapshots_dir(experiment_id: str, lane_id: str) -> Path:
    """Get the snapshots directory for a lane."""
    return get_lane_dir(experiment_id, lane_id) / "snapshots"


def get_snapshot_dir(experiment_id: str, lane_id: str, marker: int) -> Path:
    """Get the directory for a specific snapshot."""
    return get_snapshots_dir(experiment_id, lane_id) / str(marker)


def get_terminal_snapshot_dir(experiment_id: str, lane_id: str) -> Path:
    """Get the directory for the terminal snapshot (from IRATRAIN)."""
    return get_lane_dir(experiment_id, lane_id) / "terminal_snapshot"


def get_dev_snapshots_base_dir(experiment_id: str, lane_id: str) -> Path:
    """Get base directory for development snapshots (DEVTRAIN)."""
    return get_lane_dir(experiment_id, lane_id) / "dev_snapshots"


def get_dev_snapshot_dir(experiment_id: str, lane_id: str, milestone_index: int) -> Path:
    """Get directory for specific milestone snapshot (DEVTRAIN)."""
    return get_dev_snapshots_base_dir(experiment_id, lane_id) / f"milestone_{milestone_index}"


def get_dev_metrics_dir(experiment_id: str, lane_id: str) -> Path:
    """Get metrics directory for DEVTRAIN."""
    return get_dev_snapshots_base_dir(experiment_id, lane_id) / "metrics"


def get_terminal_analysis_dir(experiment_id: str, lane_id: str) -> Path:
    """Get directory for terminal representation analysis (TERMREP)."""
    return get_lane_dir(experiment_id, lane_id) / "terminal_analysis"


def get_snapshot_analysis_dir(experiment_id: str, lane_id: str) -> Path:
    """Get directory for snapshot analysis (SNAPANALYSIS)."""
    return get_lane_dir(experiment_id, lane_id) / "snapshot_analysis"





def get_saeanalysis_dir(experiment_id: str, lane_id: str) -> Path:
    """Get directory for SAE feature analysis (SAEANALYSIS)."""
    return get_lane_dir(experiment_id, lane_id) / "sae_analysis"


def get_causal_dir(experiment_id: str, lane_id: str) -> Path:
    """Get directory for causal intervention analysis (CAUSAL)."""
    return get_lane_dir(experiment_id, lane_id) / "causal_intervention"


def get_metrics_dir(experiment_id: str, lane_id: str) -> Path:
    """Get the metrics directory for a lane."""
    return get_lane_dir(experiment_id, lane_id) / "metrics"


def get_experiment_datasets_dir(experiment_id: str) -> Path:
    """Get the datasets directory for an experiment."""
    return get_experiment_dir(experiment_id) / "datasets"


def get_dataset_dir(experiment_id: str, dataset_id: str) -> Path:
    """Get the directory for a specific dataset within an experiment."""
    return get_experiment_datasets_dir(experiment_id) / dataset_id


def get_pretrained_weights_dir(experiment_id: str) -> Path:
    """Get the pretrained weights directory for an experiment."""
    return get_experiment_dir(experiment_id) / "pretrained_weights"


def get_pretrained_weights_path(experiment_id: str, network_type: str) -> Path:
    """Get the path to pretrained weights file for a specific network."""
    return get_pretrained_weights_dir(experiment_id) / network_type / "weights.pt"


def ensure_lane_structure(experiment_id: str, lane_id: str) -> None:
    """
    Create the directory structure for a lane.

    Creates:
        - data/experiments/{experiment_id}/lanes/{lane_id}/
        - data/experiments/{experiment_id}/lanes/{lane_id}/weights/
        - data/experiments/{experiment_id}/lanes/{lane_id}/snapshots/
        - data/experiments/{experiment_id}/lanes/{lane_id}/terminal_snapshot/
        - data/experiments/{experiment_id}/lanes/{lane_id}/metrics/
    """
    weights_dir = get_weights_dir(experiment_id, lane_id)
    snapshots_dir = get_snapshots_dir(experiment_id, lane_id)
    terminal_dir = get_terminal_snapshot_dir(experiment_id, lane_id)
    metrics_dir = get_metrics_dir(experiment_id, lane_id)

    weights_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    terminal_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)


def delete_experiment_dir(experiment_id: str) -> bool:
    """
    Delete the entire directory for an experiment.

    Returns:
        True if directory existed and was deleted, False otherwise
    """
    experiment_dir = get_experiment_dir(experiment_id)
    if experiment_dir.exists():
        shutil.rmtree(experiment_dir)
        return True
    return False


def delete_lane_dir(experiment_id: str, lane_id: str) -> bool:
    """
    Delete the directory for a lane.

    Returns:
        True if directory existed and was deleted, False otherwise
    """
    lane_dir = get_lane_dir(experiment_id, lane_id)
    if lane_dir.exists():
        shutil.rmtree(lane_dir)
        return True
    return False


def get_directory_size(path: Path) -> int:
    """
    Calculate the total size of a directory in bytes.

    Args:
        path: Path to the directory

    Returns:
        Total size in bytes
    """
    total_size = 0
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    for item in path.rglob('*'):
        if item.is_file():
            try:
                total_size += item.stat().st_size
            except (OSError, PermissionError):
                # Skip files we can't access
                pass

    return total_size


def get_experiment_storage_size(experiment_id: str) -> int:
    """
    Calculate the total storage size of an experiment in bytes.

    Args:
        experiment_id: The experiment ID

    Returns:
        Total size in bytes
    """
    experiment_dir = get_experiment_dir(experiment_id)
    return get_directory_size(experiment_dir)
