"""
Development Training (DEVTRAIN) - Training with periodic snapshots
"""
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
import time
import random
import numpy as np
from typing import List, Dict, Any, Optional

from training import (
    get_dataset,
    create_optimizer,
    train_epoch,
    validate,
    extract_activations,
    extract_individual_activations,
    extract_individual_multilayer_activations,
    extract_multilayer_activations,
    train_linear_probes,
    compute_class_metrics,
    compute_random_baseline,
    _clear_memory_cache,
    RemappedDataset,
)
from models import create_network
from initialization import load_weights


def _calibrate_batchnorm(model, train_loader, device, n_batches=50):
    """Run forward passes in train mode to recalibrate BN running statistics.

    After causal perturbation/injection, BN running_mean and running_var are
    reset to zero/one.  Eval-mode BN uses these stats, so validation right
    after reset gives garbage accuracy.  This function lets BN accumulate
    proper running stats without updating model weights.
    """
    was_training = model.training
    model.train()
    with torch.no_grad():
        for i, (inputs, _) in enumerate(train_loader):
            if i >= n_batches:
                break
            inputs = inputs.to(device)
            model(inputs)
    if not was_training:
        model.eval()
    _clear_memory_cache()


def capture_snapshot(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    snapshot_dir: Path,
    netinit_config: Any,
    num_classes: int,
    milestone_index: int,
    epoch: int,
    train_acc: float,
    val_acc: float,
    train_loss: float,
    val_loss: float,
    total_weight_updates: int,
    elapsed_time: float,
    snapshot_type: str,  # 't0_baseline', 'intermediate', 'threshold', 'final'
    write_progress_fn: callable,
    check_cancellation_fn: callable = None,
    metrics_dir: Path = None,
    snapshots_info_list: list = None,
    optimizer: Any = None,
    scheduler: Any = None,
    curriculum_swap: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Helper function to capture a snapshot at a milestone.

    Args:
        curriculum_swap: Optional dict for curriculum training. When provided,
            temporarily replaces the model head and uses fine-class loaders for
            activation extraction so that representations are always in the
            fine-class label space. Keys:
                'fine_num_classes': int — number of fine classes
                'fine_train_loader': DataLoader
                'fine_val_loader': DataLoader
    """
    if check_cancellation_fn:
        check_cancellation_fn()

    # Curriculum swap: temporarily replace head & loaders for fine-class extraction
    _curriculum_saved_head = None
    if curriculum_swap is not None:
        num_classes = curriculum_swap['fine_num_classes']
        train_loader = curriculum_swap['fine_train_loader']
        val_loader = curriculum_swap['fine_val_loader']
        # Swap model head to fine-class output
        if hasattr(model, 'fc') and model.fc.out_features != num_classes:
            _curriculum_saved_head = ('fc', model.fc)
            in_feat = model.fc.in_features
            model.fc = nn.Linear(in_feat, num_classes).to(device)
        elif hasattr(model, 'heads') and hasattr(model.heads, 'head') and model.heads.head.out_features != num_classes:
            _curriculum_saved_head = ('heads.head', model.heads.head)
            in_feat = model.heads.head.in_features
            model.heads.head = nn.Linear(in_feat, num_classes).to(device)

    print(f"\n[SNAPSHOT] Capturing {snapshot_type} snapshot (milestone {milestone_index}) at epoch {epoch}")
    write_progress_fn({
        'status': 'capturing_snapshot',
        'current_epoch': epoch,
        'milestone_index': milestone_index,
        'snapshot_type': snapshot_type,
    })

    # Create snapshot directory
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    except (FileNotFoundError, PermissionError, OSError):
        # Directory may have been deleted by reset
        if check_cancellation_fn:
            check_cancellation_fn()
        raise

    # Save checkpoint
    checkpoint_path = snapshot_dir / "checkpoint.pt"
    try:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"[SNAPSHOT] Saved checkpoint to {checkpoint_path}")
    except (FileNotFoundError, PermissionError, OSError):
        # Files may have been deleted by reset
        if check_cancellation_fn:
            check_cancellation_fn()
        raise

    # Save optimizer and scheduler state for training continuation experiments
    if optimizer is not None:
        try:
            training_state = {"optimizer": optimizer.state_dict(), "epoch": epoch}
            if scheduler is not None:
                training_state["scheduler"] = scheduler.state_dict()
            torch.save(training_state, snapshot_dir / "training_state.pt")
        except (FileNotFoundError, PermissionError, OSError):
            pass  # Non-critical: don't fail snapshot if this fails

    # Extract representations based on configured methods
    for method in netinit_config.representation_methods:
        # Check for cancellation before each extraction
        if check_cancellation_fn:
            check_cancellation_fn()

        method_value = method.value if hasattr(method, 'value') else method

        write_progress_fn({
            'status': 'extracting_representations',
            'milestone_index': milestone_index,
            'extraction_method': method_value,
            'current_epoch': epoch,
        })

        if method_value == 'mean_activations':
            print(f"[SNAPSHOT] Extracting mean activations...")
            mean_acts = extract_activations(model, val_loader, device, num_classes)
            torch.save(mean_acts, snapshot_dir / "mean_activations.pt")
            print(f"[SNAPSHOT] Saved mean activations")
            del mean_acts
            gc.collect()
            _clear_memory_cache()

        elif method_value == 'individual_activations':
            print(f"[SNAPSHOT] Extracting individual activations...")
            samples_per_class = netinit_config.snapshot_policy.samples_per_class
            individual_acts = extract_individual_activations(
                model, val_loader, device, num_classes, samples_per_class
            )
            torch.save(individual_acts, snapshot_dir / "individual_activations.pt")
            print(f"[SNAPSHOT] Saved individual activations")
            del individual_acts
            gc.collect()
            _clear_memory_cache()

        elif method_value == 'multilayer_activations':
            print(f"[SNAPSHOT] Extracting multilayer activations...")
            use_disk = False
            temp_dir = None
            if hasattr(netinit_config, 'memory_policy') and netinit_config.memory_policy:
                use_disk = netinit_config.memory_policy.disk_based_multilayer
                if use_disk:
                    temp_dir = snapshot_dir / "temp_multilayer"
            multilayer_acts = extract_multilayer_activations(
                model, val_loader, device, num_classes,
                use_disk=use_disk, temp_dir=temp_dir
            )
            torch.save(multilayer_acts, snapshot_dir / "multilayer_activations.pt")
            print(f"[SNAPSHOT] Saved multilayer activations ({len(multilayer_acts)} layers)")
            del multilayer_acts
            gc.collect()
            _clear_memory_cache()

        elif method_value == 'individual_multilayer_activations':
            print(f"[SNAPSHOT] Extracting individual multilayer activations...")
            samples_per_class = netinit_config.snapshot_policy.samples_per_class
            individual_multi = extract_individual_multilayer_activations(
                model, val_loader, device, num_classes, samples_per_class
            )
            torch.save(individual_multi, snapshot_dir / "individual_multilayer_activations.pt")
            print(f"[SNAPSHOT] Saved individual multilayer activations ({len(individual_multi)} layers)")
            del individual_multi
            gc.collect()
            _clear_memory_cache()

        elif method_value == 'linear_probing':
            print(f"[SNAPSHOT] Training linear probes...")
            # Check if we should use disk-based linear probes
            use_disk = False
            temp_dir = None
            if hasattr(netinit_config, 'memory_policy'):
                use_disk = netinit_config.memory_policy.disk_based_linear_probes
                if use_disk:
                    temp_dir = snapshot_dir / "temp_linear_probes"

            probes = train_linear_probes(
                model, train_loader, val_loader, device, num_classes,
                use_disk=use_disk, temp_dir=temp_dir
            )
            torch.save(probes, snapshot_dir / "linear_probes.pt")
            print(f"[SNAPSHOT] Saved linear probes (best val acc: {probes.get('best_val_accuracy', 0.0):.2f}%)")
            del probes
            gc.collect()
            _clear_memory_cache()

    # Extract mean multilayer activations if not already done
    # (only needed for REPMETRICS — skip when representation_methods is empty,
    #  e.g. causal continuation runs that only need individual_multilayer for SAE)
    rep_methods = getattr(netinit_config, 'representation_methods', None) or []
    if rep_methods:
        multilayer_file = snapshot_dir / "multilayer_activations.pt"
        if not multilayer_file.exists():
            print(f"[SNAPSHOT] Extracting multilayer activations (for REPMETRICS)...")
            if check_cancellation_fn:
                check_cancellation_fn()
            write_progress_fn({
                'status': 'extracting_representations',
                'milestone_index': milestone_index,
                'extraction_method': 'multilayer_activations',
                'current_epoch': epoch,
            })
            use_disk = False
            temp_dir = None
            if hasattr(netinit_config, 'memory_policy') and netinit_config.memory_policy:
                use_disk = netinit_config.memory_policy.disk_based_multilayer
                if use_disk:
                    temp_dir = snapshot_dir / "temp_multilayer"
            multilayer_acts = extract_multilayer_activations(
                model, val_loader, device, num_classes,
                use_disk=use_disk, temp_dir=temp_dir
            )
            torch.save(multilayer_acts, multilayer_file)
            print(f"[SNAPSHOT] Saved multilayer activations ({len(multilayer_acts)} layers)")
            del multilayer_acts
            gc.collect()
            _clear_memory_cache()

    # Always extract individual multilayer activations if not already done
    # (required by SAEANALYSIS for per-layer SAE training)
    ind_multilayer_file = snapshot_dir / "individual_multilayer_activations.pt"
    if not ind_multilayer_file.exists():
        print(f"[SNAPSHOT] Extracting individual multilayer activations (for SAEANALYSIS)...")
        if check_cancellation_fn:
            check_cancellation_fn()
        write_progress_fn({
            'status': 'extracting_representations',
            'milestone_index': milestone_index,
            'extraction_method': 'individual_multilayer_activations',
            'current_epoch': epoch,
        })
        samples_per_class = netinit_config.snapshot_policy.samples_per_class
        individual_multi = extract_individual_multilayer_activations(
            model, val_loader, device, num_classes, samples_per_class
        )
        torch.save(individual_multi, ind_multilayer_file)
        print(f"[SNAPSHOT] Saved individual multilayer activations ({len(individual_multi)} layers)")
        del individual_multi
        gc.collect()
        _clear_memory_cache()

    # Always save per-sample predictions (for SAEANALYSIS behavioral coupling)
    predictions_file = snapshot_dir / "sample_predictions.pt"
    if not predictions_file.exists():
        print(f"[SNAPSHOT] Computing per-sample predictions...")
        if check_cancellation_fn:
            check_cancellation_fn()
        was_training = model.training
        model.eval()
        sample_predictions: Dict[int, Dict[str, list]] = {}
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                for i in range(len(labels)):
                    class_idx = labels[i].item()
                    if class_idx not in sample_predictions:
                        sample_predictions[class_idx] = {
                            'predictions': [], 'correct': [], 'confidences': []
                        }
                    sample_predictions[class_idx]['predictions'].append(preds[i].item())
                    sample_predictions[class_idx]['correct'].append(
                        (preds[i] == labels[i]).item()
                    )
                    sample_predictions[class_idx]['confidences'].append(
                        probs[i].max().item()
                    )
        # Convert lists to tensors for compact storage
        for class_idx in sample_predictions:
            for key in sample_predictions[class_idx]:
                sample_predictions[class_idx][key] = torch.tensor(
                    sample_predictions[class_idx][key]
                )
        torch.save(sample_predictions, predictions_file)
        print(f"[SNAPSHOT] Saved per-sample predictions ({len(sample_predictions)} classes)")
        if was_training:
            model.train()
        del sample_predictions
        gc.collect()
        _clear_memory_cache()

    # Create snapshot metadata
    snapshot_info = {
        'milestone_index': milestone_index,
        'snapshot_type': snapshot_type,
        'is_t0_baseline': snapshot_type == 't0_baseline',
        'epoch': epoch,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'weight_updates': total_weight_updates,
        'elapsed_time_seconds': elapsed_time,
        'snapshot_dir': str(snapshot_dir),
    }

    # Write snapshots file immediately so frontend can see progress
    # This allows the timeline to update in real-time during training
    # Include the current snapshot_info in the write (caller appends after return,
    # so snapshots_info_list doesn't contain this snapshot yet)
    if metrics_dir and snapshots_info_list is not None:
        try:
            snapshots_file = metrics_dir / "devtrain_snapshots.json"
            with open(snapshots_file, 'w') as f:
                json.dump(snapshots_info_list + [snapshot_info], f, indent=2)
            print(f"[SNAPSHOT] Updated snapshots file: {len(snapshots_info_list) + 1} snapshots")
        except (FileNotFoundError, PermissionError, OSError):
            # Directory may have been deleted by reset
            if check_cancellation_fn:
                check_cancellation_fn()
            raise
        except Exception as e:
            print(f"[WARNING] Failed to write snapshots file: {e}")

    # Restore curriculum head if swapped
    if _curriculum_saved_head is not None:
        attr_path, saved_module = _curriculum_saved_head
        if attr_path == 'fc':
            model.fc = saved_module
        elif attr_path == 'heads.head':
            model.heads.head = saved_module

    print(f"[SNAPSHOT] Milestone {milestone_index} complete\n")
    return snapshot_info


def calculate_milestone_accuracies(
    num_milestones: int,
    distribution: str,  # 'uniform', 'dense_early', 'dense_late'
    terminal_accuracy: float,
    random_baseline: float = 0.0,
    baseline_margin: float = 5.0,
) -> List[float]:
    """
    Calculate target accuracy values for intermediate milestones.

    Args:
        num_milestones: Number of intermediate milestones (excluding terminal)
        distribution: Distribution scheme ('uniform', 'dense_early', 'dense_late')
        terminal_accuracy: Target terminal accuracy (0-100)
        random_baseline: Random baseline accuracy for the task (0-100)
        baseline_margin: Margin above random baseline for first milestone (%)

    Returns:
        Sorted list of accuracy milestones in ascending order, all above (random_baseline + baseline_margin)
    """
    if num_milestones == 0:
        return []

    # Ensure milestones start above random baseline
    min_milestone = random_baseline + baseline_margin
    milestone_range = terminal_accuracy - min_milestone

    # Ensure we have positive range
    if milestone_range <= 0:
        # If terminal accuracy is too close to baseline, distribute evenly anyway
        milestone_range = max(1.0, terminal_accuracy - random_baseline)
        min_milestone = random_baseline

    milestones = []

    if distribution == 'uniform':
        # Evenly spaced accuracy milestones
        for i in range(num_milestones):
            ratio = (i + 1) / (num_milestones + 1)
            milestones.append(min_milestone + milestone_range * ratio)

    elif distribution == 'dense_early':
        # More snapshots early in training (power curve)
        # Power curve clusters ratios near 0 = low accuracies = early in training
        power = 2.5  # Controls the curve steepness
        for i in range(num_milestones):
            ratio = (i + 1) / (num_milestones + 1)
            milestones.append(min_milestone + milestone_range * (ratio ** power))

    elif distribution == 'dense_late':
        # More snapshots late in training (saturating exponential)
        # Exponential curve clusters ratios near 1.0 = high accuracies = late in training
        max_x = 5  # Controls the spread
        for i in range(num_milestones):
            x = ((i + 1) * max_x) / (num_milestones + 1)
            ratio = (1 - (2.718281828459045 ** (-x))) / (1 - (2.718281828459045 ** (-max_x)))
            milestones.append(min_milestone + milestone_range * ratio)

    else:
        # Default to uniform
        for i in range(num_milestones):
            ratio = (i + 1) / (num_milestones + 1)
            milestones.append(min_milestone + milestone_range * ratio)

    return milestones


def calculate_milestone_weight_updates(
    num_milestones: int,
    distribution: str,  # 'uniform', 'dense_early', 'dense_late'
    total_weight_updates: int,
) -> List[int]:
    """
    Calculate weight-update (batch count) targets for intermediate milestones.

    Milestones are distributed across [min_wu, total_weight_updates] using the
    same distribution schemes as accuracy milestones, but in the weight-update
    domain.  This guarantees network-independent snapshot timing.

    Args:
        num_milestones: Number of intermediate milestones
        distribution: Distribution scheme ('uniform', 'dense_early', 'dense_late')
        total_weight_updates: Total batch count for full training

    Returns:
        Sorted list of absolute weight-update counts for milestones
    """
    import math

    if num_milestones == 0 or total_weight_updates <= 0:
        return []

    # Reserve ~1% at the start
    min_wu = max(1, int(total_weight_updates * 0.01))
    wu_range = total_weight_updates - min_wu
    milestones = []

    if distribution == 'uniform':
        for i in range(num_milestones):
            ratio = (i + 1) / (num_milestones + 1)
            milestones.append(int(min_wu + wu_range * ratio))
    elif distribution == 'dense_early':
        # Power curve — clusters near start (low WU counts = early training)
        # Matches calculate_milestone_accuracies dense_early behavior
        power = 2.5
        for i in range(num_milestones):
            ratio = (i + 1) / (num_milestones + 1)
            milestones.append(int(min_wu + wu_range * (ratio ** power)))
    elif distribution == 'dense_late':
        # Exponential saturation — clusters near end (high WU counts = late training)
        # Matches calculate_milestone_accuracies dense_late behavior
        max_x = 5
        for i in range(num_milestones):
            x = ((i + 1) * max_x) / (num_milestones + 1)
            ratio = (1 - math.exp(-x)) / (1 - math.exp(-max_x))
            milestones.append(int(min_wu + wu_range * ratio))
    else:
        # Default to uniform
        for i in range(num_milestones):
            ratio = (i + 1) / (num_milestones + 1)
            milestones.append(int(min_wu + wu_range * ratio))

    return milestones


def calculate_intermediate_milestones(
    total_epochs: int,
    num_intermediate: int,
    distribution: str,  # 'uniform', 'dense_early', 'dense_late'
) -> List[int]:
    """
    [DEPRECATED] Calculate intermediate milestone epochs - USE ACCURACY-BASED MILESTONES INSTEAD.
    This function is kept for reference but is no longer used in the main training loop.

    Args:
        total_epochs: Total number of epochs to train for
        num_intermediate: Number of intermediate milestones
        distribution: Distribution scheme ('uniform', 'dense_early', 'dense_late')

    Returns:
        Sorted list of epoch numbers for intermediate milestones (NOT starting at T=0)
    """
    if num_intermediate == 0:
        return []

    if distribution == 'uniform':
        # Evenly spaced throughout training, not starting at 0
        # Example: [20, 40, 60, 80] for 100 epochs, 4 intermediate milestones
        return [int((i + 1) * total_epochs / (num_intermediate + 1)) for i in range(num_intermediate)]

    elif distribution == 'dense_early':
        # More snapshots early: quadratic spacing
        # Map positions from 0 to 1 (excluding 0), square them to get early density
        positions = [(i / (num_intermediate + 1)) ** 2 for i in range(1, num_intermediate + 1)]
        return [max(1, int(pos * total_epochs)) for pos in positions]

    elif distribution == 'dense_late':
        # More snapshots late: inverse quadratic
        positions = [1 - ((num_intermediate + 1 - i) / (num_intermediate + 1)) ** 2 for i in range(1, num_intermediate + 1)]
        return [max(1, int(pos * total_epochs)) for pos in positions]

    else:
        # Default to uniform
        return [int((i + 1) * total_epochs / (num_intermediate + 1)) for i in range(num_intermediate)]


def flush_metrics_to_disk(
    metrics_history: List[Dict],
    metrics_file: Path,
    keep_recent: int = 100
) -> List[Dict]:
    """
    Flush older metrics to disk and return recent metrics to keep in RAM.

    Args:
        metrics_history: Current metrics history in RAM
        metrics_file: File to append flushed metrics to
        keep_recent: Number of recent epochs to keep in RAM

    Returns:
        Truncated metrics_history with only recent entries
    """
    if len(metrics_history) <= keep_recent:
        return metrics_history

    # Split into old (to flush) and recent (to keep)
    to_flush = metrics_history[:-keep_recent]
    to_keep = metrics_history[-keep_recent:]

    # Append flushed metrics to disk file (JSONL format for incremental appending)
    metrics_jsonl = metrics_file.with_suffix('.jsonl')
    try:
        with open(metrics_jsonl, 'a') as f:
            for metric in to_flush:
                f.write(json.dumps(metric) + '\n')
        print(f"[METRICS] Flushed {len(to_flush)} epochs to disk, keeping {len(to_keep)} in RAM")
    except Exception as e:
        print(f"[WARNING] Failed to flush metrics to disk: {e}")
        # If flush fails, keep all metrics in RAM
        return metrics_history

    return to_keep


def train_with_snapshots(
    initial_weights_path: str,
    netinit_config: Any,  # NetInitConfiguration object
    dataset_id: str,
    selected_classes: List[str],
    snapshots_base_dir: str,  # /dev_snapshots/ directory
    metrics_dir: str,
    experiment_id: str,
    stove_id: Optional[str] = None,
    resume_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Train network with periodic snapshot capture at milestones (DEVTRAIN).

    Args:
        initial_weights_path: Path to initial weights from NETINIT
        netinit_config: NetInitConfiguration object
        dataset_id: Dataset identifier
        selected_classes: List of selected class names
        snapshots_base_dir: Base directory for dev snapshots
        metrics_dir: Directory to save training metrics
        experiment_id: Experiment ID for locating experiment-specific dataset
        stove_id: Optional identifier for progress file updates
        resume_state: Optional dict for resuming interrupted training with keys:
            - checkpoint_path: Path to model weights to resume from
            - start_epoch: Epoch to resume from (1-based)
            - existing_snapshots: List of existing snapshot info dicts
            - next_milestone_index: Next milestone index to use
            - milestones_reached: List of bools for which milestones are reached

    Returns:
        Summary info dict with total_snapshots, final_epoch, etc.
    """
    is_resuming = resume_state is not None

    # Create directories
    Path(snapshots_base_dir).mkdir(parents=True, exist_ok=True)
    Path(metrics_dir).mkdir(parents=True, exist_ok=True)

    progress_file = Path(metrics_dir) / "devtrain_progress.json"
    cancellation_file = Path(metrics_dir) / "devtrain_cancel.flag"

    # Remove any existing cancellation flag at start; only clear progress on fresh start
    if cancellation_file.exists():
        cancellation_file.unlink()
    if not is_resuming and progress_file.exists():
        progress_file.unlink()

    def check_cancellation():
        """Check if training has been cancelled via reset"""
        if cancellation_file.exists():
            print("\n[CANCELLATION] Training cancelled by user (reset requested)")
            raise Exception("Training cancelled by user")

    # Accumulate initialization steps so the backend polling doesn't miss any
    _init_steps_log: List[Dict[str, str]] = []

    def log_initialization_step(step: str, detail: str = ""):
        """Log and save initialization progress"""
        message = f"[INIT] {step}"
        if detail:
            message += f": {detail}"
        print(message)

        if stove_id:
            _init_steps_log.append({'step': step, 'detail': detail})
            init_data = {
                'initialization_step': step,
                'initialization_detail': detail,
                'initialization_steps': _init_steps_log,
                'status': 'initializing',
                'current_epoch': 0,
            }
            try:
                with open(progress_file, 'w') as f:
                    json.dump(init_data, f, indent=2)
            except Exception:
                pass

    def write_progress(data: dict):
        """Write progress updates to file for WebSocket polling"""
        if stove_id:
            try:
                with open(progress_file, 'w') as f:
                    json.dump(data, f, indent=2)
            except (FileNotFoundError, PermissionError, OSError):
                # Files may have been deleted by reset - check for cancellation
                check_cancellation()
            except Exception:
                pass

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    log_initialization_step("Device setup", f"Using {device}")

    # Set random seed for reproducibility
    if netinit_config.random_seed is not None:
        log_initialization_step("Setting random seed", f"seed={netinit_config.random_seed}")
        torch.manual_seed(netinit_config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(netinit_config.random_seed)
        np.random.seed(netinit_config.random_seed)
        random.seed(netinit_config.random_seed)

    # Load dataset
    log_initialization_step("Loading dataset", f"{dataset_id} with {len(selected_classes)} classes")
    transform_cfg = netinit_config.transform_config.model_dump() if netinit_config.transform_config else None
    train_dataset, class_names = get_dataset(dataset_id, selected_classes, train=True, experiment_id=experiment_id, transform_config=transform_cfg)
    val_dataset, _ = get_dataset(dataset_id, selected_classes, train=False, experiment_id=experiment_id, transform_config=transform_cfg)
    log_initialization_step("Dataset loaded", f"{len(train_dataset)} training samples")

    # MPS does not support multiprocessing DataLoader workers reliably
    dl_workers = 0 if device == "mps" else 2
    # Cap batch size on MPS to avoid OOM with large models (e.g. ViT-B16)
    batch_size = netinit_config.batch_size
    if device == "mps" and batch_size > 16:
        print(f"[MPS] Reducing batch size from {batch_size} to 16 to avoid memory pressure")
        batch_size = 16
    log_initialization_step("Creating data loaders", f"batch_size={batch_size}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=dl_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dl_workers
    )

    # ── Label noise policy setup ──────────────────────────────────────
    noise_policy = getattr(netinit_config, 'noise_policy', None)
    if noise_policy and getattr(noise_policy, 'enabled', False):
        from training import NoisyLabelDataset
        from cifar100_hierarchy import build_superclass_map

        # Build superclass map: class_idx → superclass_name
        class_to_idx = {name: i for i, name in enumerate(selected_classes)}
        sc_map = build_superclass_map(selected_classes, class_to_idx)

        noisy_train = NoisyLabelDataset(
            base_dataset=train_dataset,
            noise_type=noise_policy.noise_type,
            noise_prob=noise_policy.noise_prob,
            superclass_map=sc_map,
            num_classes=len(selected_classes),
        )
        train_loader = DataLoader(
            noisy_train, batch_size=batch_size,
            shuffle=True, num_workers=dl_workers,
        )
        print(f"[NOISE] Label noise enabled: type={noise_policy.noise_type}, "
              f"p={noise_policy.noise_prob}")

    # ── Curriculum policy setup ─────────────────────────────────────────
    curriculum_policy = getattr(netinit_config, 'curriculum_policy', None)
    curriculum_enabled = curriculum_policy is not None and getattr(curriculum_policy, 'enabled', False)
    curriculum_phases = []
    superclass_label_map = None  # fine_class_idx → superclass_idx
    superclass_names = None
    num_superclasses = None
    curriculum_current_phase = None  # Track current phase to detect transitions
    # Saved head weights + optimizer state per label_mode for alternating curricula.
    # When switching away from a mode, we stash {head_state_dict, optimizer_state_dict}
    # so that switching back restores accumulated momentum / weight history.
    _curriculum_head_cache = {}  # label_mode -> {'head': state_dict, 'optimizer': state_dict}
    # Keep fine-class loaders for snapshot capture (representations must always be
    # extracted in the fine-class label space so TERMREP/SAEANALYSIS stay consistent)
    fine_train_loader = train_loader
    fine_val_loader = val_loader

    if curriculum_enabled:
        phases_raw = getattr(curriculum_policy, 'phases', [])
        if hasattr(phases_raw, '__iter__'):
            for p in phases_raw:
                phase = {
                    'start_epoch': getattr(p, 'start_epoch', None) or p.get('start_epoch'),
                    'end_epoch': getattr(p, 'end_epoch', None) or p.get('end_epoch'),
                    'label_mode': getattr(p, 'label_mode', None) or p.get('label_mode'),
                    'learning_rate': getattr(p, 'learning_rate', None) if hasattr(p, 'learning_rate') else p.get('learning_rate'),
                } if not isinstance(p, dict) else p
                curriculum_phases.append(phase)

        if curriculum_phases:
            log_initialization_step("Curriculum policy enabled",
                f"{len(curriculum_phases)} phases: " +
                ", ".join(f"ep{p['start_epoch']}-{p['end_epoch']}:{p['label_mode']}" for p in curriculum_phases))

            # Build superclass label mapping for 'superclass' phases
            # Import hierarchy builder based on dataset
            if dataset_id in ('cifar100',):
                from cifar100_hierarchy import CIFAR100_SUPERCLASSES
                superclass_names = sorted(CIFAR100_SUPERCLASSES.keys())
                super_to_idx = {name: i for i, name in enumerate(superclass_names)}
                fine_to_super = {}
                for sname, fines in CIFAR100_SUPERCLASSES.items():
                    for fname in fines:
                        fine_to_super[fname] = super_to_idx[sname]
                # Map fine-class index (as used in the dataset) to superclass index
                superclass_label_map = {}
                for fine_idx, fine_name in enumerate(selected_classes):
                    bare = fine_name.split("_", 1)[1] if fine_name.startswith(f"{dataset_id}_") else fine_name
                    superclass_label_map[fine_idx] = fine_to_super.get(bare, fine_to_super.get(fine_name, 0))
                num_superclasses = len(superclass_names)
                log_initialization_step("Superclass mapping built",
                    f"{len(selected_classes)} fine classes → {num_superclasses} superclasses")
            else:
                log_initialization_step("WARNING", f"Curriculum superclass mode not supported for dataset {dataset_id}")
                curriculum_enabled = False

    # Create model
    num_classes = len(selected_classes)

    # Compute random baseline from dataset
    log_initialization_step("Computing random baseline", f"for {num_classes}-class task")
    baseline_info = compute_random_baseline(train_dataset, num_classes)
    random_baseline = baseline_info['random_baseline']
    class_distribution = baseline_info['class_distribution']
    class_counts = baseline_info['class_counts']
    is_balanced = baseline_info['is_balanced']

    log_initialization_step(
        "Random baseline computed",
        f"{random_baseline:.2f}% ({'balanced' if is_balanced else 'imbalanced'} dataset)"
    )

    # Validate accuracy threshold
    accuracy_threshold = netinit_config.training_policy.accuracy_threshold or 95.0
    if accuracy_threshold <= random_baseline:
        print(f"[WARNING] Accuracy threshold ({accuracy_threshold:.2f}%) is below or equal to random baseline ({random_baseline:.2f}%)!")
        print(f"[WARNING] This threshold may never be reached through learning.")

    log_initialization_step("Creating network", f"{netinit_config.network_type} for {num_classes} classes")
    model = create_network(
        network_type=netinit_config.network_type,
        num_classes=num_classes,
        pretrained=False
    )

    # Load weights - from checkpoint if resuming, from initial weights otherwise
    if is_resuming:
        resume_checkpoint_path = resume_state['checkpoint_path']
        log_initialization_step("Loading weights from checkpoint", f"resuming from {Path(resume_checkpoint_path).parent.name}")
        load_weights(model, resume_checkpoint_path, device=device)
    else:
        log_initialization_step("Loading initial weights", "from NETINIT")
        load_weights(model, initial_weights_path, device=device)

    log_initialization_step("Moving model to device", device)
    model = model.to(device)

    # Create optimizer
    log_initialization_step("Creating optimizer", f"{netinit_config.optimizer.type}")
    optimizer_config = netinit_config.optimizer.model_dump()
    # Strip None values so create_optimizer's .get(key, default) returns actual defaults
    # (model_dump() preserves None-valued keys, which defeats .get()'s default mechanism)
    optimizer_config = {k: v for k, v in optimizer_config.items() if v is not None}
    optimizer = create_optimizer(model, optimizer_config)

    # Calculate snapshot milestones according to policy
    total_epochs = netinit_config.training_policy.epochs or 100  # Default to 100 if None
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # When resuming, restore optimizer/scheduler state if available, otherwise step scheduler forward
    if is_resuming:
        resume_start_epoch = resume_state['start_epoch']
        training_state_path = resume_state.get('training_state_path')
        if training_state_path and Path(training_state_path).exists():
            log_initialization_step("Restoring optimizer/scheduler state", f"from {Path(training_state_path).name}")
            training_state = torch.load(training_state_path, map_location=device, weights_only=False)
            optimizer.load_state_dict(training_state["optimizer"])
            if "scheduler" in training_state:
                scheduler.load_state_dict(training_state["scheduler"])
            else:
                for _ in range(resume_start_epoch):
                    scheduler.step()
        else:
            log_initialization_step("Advancing scheduler", f"stepping to epoch {resume_start_epoch}")
            for _ in range(resume_start_epoch):
                scheduler.step()
    terminal_capture = netinit_config.snapshot_policy.terminal_capture.value if hasattr(netinit_config.snapshot_policy.terminal_capture, 'value') else netinit_config.snapshot_policy.terminal_capture
    # Terminal capture modes:
    #   'at_threshold': Stop training when accuracy threshold is reached (early stopping)
    #   'final_epoch': Always train to final epoch, regardless of threshold
    #   'both': Stop at whichever comes first - threshold reached OR final epoch reached
    distribution = netinit_config.snapshot_policy.distribution_scheme.value if hasattr(netinit_config.snapshot_policy.distribution_scheme, 'value') else netinit_config.snapshot_policy.distribution_scheme
    accuracy_threshold = netinit_config.training_policy.accuracy_threshold or 95.0  # Default to 95% if None

    baseline_margin = 5.0  # Margin above random baseline for first milestone

    # Detect milestone type (accuracy or weight_updates)
    milestone_type_raw = getattr(netinit_config.snapshot_policy, 'milestone_type', 'accuracy')
    milestone_type = milestone_type_raw.value if hasattr(milestone_type_raw, 'value') else milestone_type_raw

    # Weight-update milestone tracking (empty when in accuracy mode)
    intermediate_milestone_weight_updates: List[int] = []
    milestone_wu_reached: List[bool] = []

    if milestone_type == 'weight_updates':
        # --- Weight-update mode ---
        # Compute total weight updates from actual data loader size (runtime value
        # accounts for MPS batch-size cap and dataset rounding)
        total_wu = total_epochs * len(train_loader)

        # Use pre-calculated milestones from NETINIT config, or compute at runtime
        pre_calculated = getattr(netinit_config.snapshot_policy, 'milestone_weight_updates', None)
        if pre_calculated:
            intermediate_milestone_weight_updates = list(pre_calculated)
            log_initialization_step(
                "Using weight-update milestones from NETINIT",
                f"WU milestones: {intermediate_milestone_weight_updates}"
            )
        else:
            intermediate_milestone_weight_updates = calculate_milestone_weight_updates(
                num_milestones=netinit_config.snapshot_policy.milestone_count,
                distribution=distribution,
                total_weight_updates=total_wu,
            )
            log_initialization_step(
                "Calculated weight-update milestones at runtime",
                f"total_wu={total_wu}, milestones: {intermediate_milestone_weight_updates}"
            )

        milestone_wu_reached = [False] * len(intermediate_milestone_weight_updates)
        # Empty accuracy milestones so accuracy-based checks are skipped
        intermediate_milestone_accuracies = []

        log_initialization_step(
            "Snapshot policy configured (weight-update mode)",
            f"T=0: WU=0, Intermediate ({len(intermediate_milestone_weight_updates)}): {intermediate_milestone_weight_updates}, Terminal: WU={total_wu}"
        )
    else:
        # --- Accuracy mode (original) ---
        # Get milestone accuracies from NETINIT config (calculated during NETINIT phase)
        # This ensures 1:1 consistency between what was planned and what actually happens
        if netinit_config.snapshot_policy.milestone_accuracies is not None:
            # Use pre-calculated milestones from NETINIT
            intermediate_milestone_accuracies = netinit_config.snapshot_policy.milestone_accuracies
            log_initialization_step(
                "Using milestone accuracies from NETINIT",
                f"Intermediate milestones: {[f'{acc:.1f}%' for acc in intermediate_milestone_accuracies]}"
            )
        else:
            # Fallback: Calculate milestones if not provided (backward compatibility)
            log_initialization_step(
                "Calculating milestone accuracies from snapshot policy",
                f"milestone_count={netinit_config.snapshot_policy.milestone_count}"
            )
            intermediate_milestone_accuracies = calculate_milestone_accuracies(
                num_milestones=netinit_config.snapshot_policy.milestone_count,
                distribution=distribution,
                terminal_accuracy=accuracy_threshold,
                random_baseline=random_baseline,
                baseline_margin=baseline_margin,
            )

        log_initialization_step(
            "Snapshot policy configured",
            f"T=0: {random_baseline:.2f}%, Intermediate ({len(intermediate_milestone_accuracies)}): {[f'{acc:.1f}%' for acc in intermediate_milestone_accuracies]}, Terminal: {terminal_capture} @ {accuracy_threshold:.1f}%"
        )

    # Save snapshot policy to progress file
    snapshot_policy_info = {
        'milestone_count': netinit_config.snapshot_policy.milestone_count,
        'milestone_type': milestone_type,
        'distribution_scheme': distribution,
        'terminal_capture': terminal_capture,
        'milestone_accuracies': intermediate_milestone_accuracies if milestone_type == 'accuracy' else [],
        'milestone_weight_updates': intermediate_milestone_weight_updates if milestone_type == 'weight_updates' else [],
        'random_baseline': random_baseline,
        'terminal_accuracy': accuracy_threshold,
        'baseline_margin': baseline_margin,
        'includes_t0_snapshot': True,
        'class_distribution': class_distribution,
        'class_counts': class_counts,
        'is_balanced': is_balanced,
    }
    write_progress({
        'status': 'initializing',
        'snapshot_policy': snapshot_policy_info,
    })

    # Persist snapshot policy to a dedicated file (write_progress overwrites the progress
    # file on every update, so snapshot_policy would be lost once training starts)
    snapshot_policy_file = Path(metrics_dir) / "devtrain_snapshot_policy.json"
    try:
        with open(snapshot_policy_file, 'w') as f:
            json.dump(snapshot_policy_info, f, indent=2)
    except Exception:
        pass

    # Loss function
    criterion = nn.CrossEntropyLoss()

    if is_resuming:
        # --- RESUME PATH: skip T=0 and restore state from existing snapshots ---
        log_initialization_step("Resuming training", f"from epoch {resume_state['start_epoch']} with {len(resume_state['existing_snapshots'])} existing snapshots")

        # Start tracking time
        start_time = time.time()

        # Restore snapshot tracking state
        snapshots_info = list(resume_state['existing_snapshots'])
        milestone_index = resume_state['next_milestone_index']
        milestone_accuracy_reached = list(resume_state['milestones_reached'])
        # Restore WU milestone tracking for resumed weight-update mode
        if milestone_type == 'weight_updates':
            milestone_wu_reached = list(resume_state.get('milestones_wu_reached', [False] * len(intermediate_milestone_weight_updates)))

        # Initialize metrics tracking for resumed training
        metrics_history = []
        total_weight_updates = resume_state.get('total_weight_updates', 0)
        threshold_snapshot_taken = False

        # Get current validation metrics
        check_cancellation()

        latest_val_loss, latest_val_accuracy = validate(model, val_loader, criterion, device)
        _clear_memory_cache()

        # Optionally capture T=0 baseline snapshot before training begins
        # (e.g., for causal intervention experiments that need a post-perturbation baseline)
        if resume_state.get('capture_t0', False):
            initial_train_loss, initial_train_acc = validate(model, train_loader, criterion, device)
            _clear_memory_cache()
            t0_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"
            print(f"\n[T=0 RESUME] Capturing post-resume baseline at milestone_{milestone_index}")
            print(f"  Val accuracy: {latest_val_accuracy:.2f}%, Train accuracy: {initial_train_acc:.2f}%")
            t0_info = capture_snapshot(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                snapshot_dir=t0_dir,
                netinit_config=netinit_config,
                num_classes=num_classes,
                milestone_index=milestone_index,
                epoch=resume_state['start_epoch'],
                train_acc=initial_train_acc,
                val_acc=latest_val_accuracy,
                train_loss=initial_train_loss,
                val_loss=latest_val_loss,
                total_weight_updates=total_weight_updates,
                elapsed_time=0.0,
                snapshot_type='t0_resume',
                write_progress_fn=write_progress,
                check_cancellation_fn=check_cancellation,
                metrics_dir=Path(metrics_dir),
                snapshots_info_list=snapshots_info,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            snapshots_info.append(t0_info)
            milestone_index += 1
            _clear_memory_cache()

        print(f"\n[RESUME] Continuing from epoch {resume_state['start_epoch']}")
        print(f"  Existing snapshots: {len(snapshots_info)}")
        print(f"  Next milestone index: {milestone_index}")
        print(f"  Current val accuracy: {latest_val_accuracy:.2f}%")
        print(f"  Milestones reached: {sum(milestone_accuracy_reached)}/{len(milestone_accuracy_reached)}")

    else:
        # --- FRESH START PATH: capture T=0 baseline ---
        log_initialization_step("Initialization complete", "Capturing T=0 baseline snapshot")

        # Check for cancellation before T=0 validation
        check_cancellation()

        # Validate untrained network (T=0 baseline)
        initial_val_loss, initial_val_acc = validate(model, val_loader, criterion, device)
        initial_train_loss, initial_train_acc = validate(model, train_loader, criterion, device)

        print(f"\n[T=0 BASELINE]")
        print(f"  Initial validation accuracy: {initial_val_acc:.2f}%")
        print(f"  Initial training accuracy: {initial_train_acc:.2f}%")
        print(f"  Random baseline: {random_baseline:.2f}%")
        print(f"  Expected at T=0: ~{random_baseline:.2f}% ± {baseline_margin:.1f}%")

        # Capture T=0 snapshot (milestone_0)
        t0_snapshot_dir = Path(snapshots_base_dir) / "milestone_0"
        print(f"[T=0] CAPTURING INITIAL SNAPSHOT → {t0_snapshot_dir}")

        # Start tracking time after T=0 validation and snapshot
        start_time = time.time()

        # Initialize snapshots info list (will be passed to capture_snapshot for real-time updates)
        snapshots_info = []

        t0_snapshot_info = capture_snapshot(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            snapshot_dir=t0_snapshot_dir,
            netinit_config=netinit_config,
            num_classes=num_classes,
            milestone_index=0,  # T=0 is milestone_0
            epoch=0,
            train_acc=initial_train_acc,
            val_acc=initial_val_acc,
            train_loss=initial_train_loss,
            val_loss=initial_val_loss,
            total_weight_updates=0,
            elapsed_time=0.0,
            snapshot_type='t0_baseline',
            write_progress_fn=write_progress,
            check_cancellation_fn=check_cancellation,
            metrics_dir=Path(metrics_dir),
            snapshots_info_list=snapshots_info,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        snapshots_info.append(t0_snapshot_info)

        # Clear GPU cache after T=0 snapshot
        _clear_memory_cache()

        # Recalculate milestone accuracies if T=0 accuracy is close to any milestones.
        # Only applies to accuracy mode — weight-update milestones are position-based
        # and don't depend on initial accuracy.
        # The initial milestone targets are based on the theoretical random baseline, but
        # certain initializations or architectures can start above random. Milestones
        # within t0_margin of T=0 accuracy don't represent meaningful learning progress.
        t0_margin = 3.0  # Minimum margin above T=0 accuracy for first milestone
        if intermediate_milestone_accuracies and (initial_val_acc + t0_margin) >= intermediate_milestone_accuracies[0]:
            # Count how many milestones are within t0_margin of T=0 accuracy (meaninglessly close)
            milestones_below_t0 = sum(1 for acc in intermediate_milestone_accuracies if acc <= initial_val_acc + t0_margin)

            if milestones_below_t0 > 0:
                original_milestones = list(intermediate_milestone_accuracies)
                print(f"\n[MILESTONE RECALCULATION] T=0 accuracy ({initial_val_acc:.2f}%) is at or above {milestones_below_t0} milestone target(s)")
                print(f"  Original milestones: {[f'{acc:.1f}%' for acc in original_milestones]}")

                # Recalculate all milestones between T=0 accuracy (+ margin) and terminal accuracy
                intermediate_milestone_accuracies = calculate_milestone_accuracies(
                    num_milestones=len(intermediate_milestone_accuracies),
                    distribution=distribution,
                    terminal_accuracy=accuracy_threshold,
                    random_baseline=initial_val_acc,  # Use actual T=0 accuracy as floor
                    baseline_margin=t0_margin,
                )

                print(f"  Recalculated milestones: {[f'{acc:.1f}%' for acc in intermediate_milestone_accuracies]}")
                print(f"  New range: {initial_val_acc + t0_margin:.1f}% → {accuracy_threshold:.1f}%")

                # Update snapshot policy info for dashboard
                snapshot_policy_info['milestone_accuracies'] = intermediate_milestone_accuracies
                snapshot_policy_info['milestone_recalculated'] = True
                snapshot_policy_info['original_milestone_accuracies'] = original_milestones
                snapshot_policy_info['t0_val_accuracy'] = initial_val_acc

                write_progress({
                    'status': 'initializing',
                    'snapshot_policy': snapshot_policy_info,
                })

                # Update persistent snapshot policy file with recalculated milestones
                try:
                    with open(snapshot_policy_file, 'w') as f:
                        json.dump(snapshot_policy_info, f, indent=2)
                except Exception:
                    pass

        # Initialize metrics tracking
        metrics_history = []
        milestone_index = 1  # Next snapshot will be milestone_1
        total_weight_updates = 0
        threshold_snapshot_taken = False
        milestone_accuracy_reached = [False] * len(intermediate_milestone_accuracies)  # Track which intermediate milestones have been reached
        latest_val_accuracy = initial_val_acc  # Track latest validation accuracy for progress updates
        latest_val_loss = initial_val_loss

    # Helper to check if we should do intra-epoch validation
    def should_check_intra_epoch(batch_idx: int, total_batches: int) -> bool:
        """Check if we should run validation at this batch based on policy"""
        if hasattr(netinit_config, 'validation_policy'):
            check_interval = netinit_config.validation_policy.get_check_interval(total_batches)
        else:
            # Backward compatibility: original hardcoded behavior
            check_interval = max(1, min(total_batches // 20, 10))
        return batch_idx > 0 and batch_idx % check_interval == 0

    # Helper to check if we should extract mean activations
    def should_extract_activations(batch_idx: int, total_batches: int) -> bool:
        """Extract activations periodically during training"""
        if hasattr(netinit_config, 'memory_policy') and hasattr(netinit_config.memory_policy, 'activation_extraction_enabled'):
            if not netinit_config.memory_policy.activation_extraction_enabled:
                return False
            extract_interval = netinit_config.memory_policy.activation_extraction_interval
        else:
            # Backward compatibility: original hardcoded behavior
            extract_interval = max(1, min(total_batches // 10, 20))
        return batch_idx > 0 and batch_idx % extract_interval == 0

    # Create batch log file for detailed logging (truncate on fresh start, append on resume)
    batch_log_file = Path(metrics_dir) / "devtrain_batch_log.jsonl"
    if not is_resuming and batch_log_file.exists():
        batch_log_file.unlink()

    # Determine starting epoch
    start_epoch = resume_state['start_epoch'] if is_resuming else 1

    # Initialize training state
    write_progress({
        'status': 'training',
        'current_epoch': start_epoch - 1,
        'current_milestone': milestone_index,
    })

    # Build curriculum_swap dict for snapshot capture (used when curriculum is active
    # in a superclass phase to ensure activations are extracted in fine-class label space)
    def _get_curriculum_swap():
        """Return curriculum_swap dict if currently in a superclass phase, else None."""
        if not curriculum_enabled or curriculum_current_phase is None:
            return None
        if curriculum_current_phase.get('label_mode') == 'superclass':
            return {
                'fine_num_classes': num_classes,
                'fine_train_loader': fine_train_loader,
                'fine_val_loader': fine_val_loader,
            }
        return None

    for epoch in range(start_epoch, total_epochs + 1):
        # Check for cancellation at start of each epoch
        check_cancellation()

        epoch_start = time.time()
        epoch_snapshot_taken = False  # Track if snapshot taken during this epoch

        # ── Curriculum phase switching ────────────────────────────────
        if curriculum_enabled and curriculum_phases:
            # Find which phase this epoch belongs to
            new_phase = None
            for phase in curriculum_phases:
                if phase['start_epoch'] <= epoch <= phase['end_epoch']:
                    new_phase = phase
                    break

            if new_phase and new_phase != curriculum_current_phase:
                prev_mode = curriculum_current_phase['label_mode'] if curriculum_current_phase else 'none'
                new_mode = new_phase['label_mode']
                print(f"\n[CURRICULUM] Phase transition at epoch {epoch}: {prev_mode} → {new_mode}")

                # ── Save outgoing head + optimizer state ──────────────
                def _get_head_module():
                    if hasattr(model, 'fc'):
                        return model.fc
                    elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                        return model.heads.head
                    elif hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Sequential):
                            return model.classifier[-1]
                        elif isinstance(model.classifier, nn.Linear):
                            return model.classifier
                    return None

                def _set_head_module(new_head):
                    if hasattr(model, 'fc'):
                        model.fc = new_head
                    elif hasattr(model, 'heads') and hasattr(model.heads, 'head'):
                        model.heads.head = new_head
                    elif hasattr(model, 'classifier'):
                        if isinstance(model.classifier, nn.Sequential):
                            model.classifier[-1] = new_head
                        elif isinstance(model.classifier, nn.Linear):
                            model.classifier = new_head

                if prev_mode != 'none':
                    head_mod = _get_head_module()
                    if head_mod is not None:
                        import copy
                        _curriculum_head_cache[prev_mode] = {
                            'head': copy.deepcopy(head_mod.state_dict()),
                            'optimizer': copy.deepcopy(optimizer.state_dict()),
                        }
                        print(f"[CURRICULUM] Cached head + optimizer state for '{prev_mode}' mode")

                # ── Determine target num_classes for new mode ─────────
                target_classes = num_superclasses if new_mode == 'superclass' else num_classes

                if new_mode == 'superclass' and superclass_label_map is not None:
                    # Switch to superclass labels (use module-level class for picklability with num_workers>0)
                    train_dataset_super = RemappedDataset(train_dataset, superclass_label_map)
                    val_dataset_super = RemappedDataset(val_dataset, superclass_label_map)
                    train_loader = DataLoader(train_dataset_super, batch_size=batch_size, shuffle=True, num_workers=dl_workers)
                    val_loader = DataLoader(val_dataset_super, batch_size=batch_size, shuffle=False, num_workers=dl_workers)

                elif new_mode == 'fine':
                    # Switch (back) to fine-class labels
                    train_loader = fine_train_loader
                    val_loader = fine_val_loader

                # ── Restore or create head ────────────────────────────
                head_mod = _get_head_module()
                in_features = head_mod.in_features if head_mod is not None else None

                if new_mode in _curriculum_head_cache:
                    # Restore previously cached head weights
                    new_head = nn.Linear(in_features, target_classes).to(device)
                    new_head.load_state_dict(_curriculum_head_cache[new_mode]['head'])
                    _set_head_module(new_head)
                    print(f"[CURRICULUM] Restored cached head weights for '{new_mode}' mode")
                else:
                    # First time entering this mode — fresh init
                    new_head = nn.Linear(in_features, target_classes).to(device)
                    nn.init.kaiming_normal_(new_head.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.zeros_(new_head.bias)
                    _set_head_module(new_head)

                # ── Restore or create optimizer ───────────────────────
                optimizer_config_dict = netinit_config.optimizer.model_dump() if hasattr(netinit_config.optimizer, 'model_dump') else dict(netinit_config.optimizer.__dict__)
                optimizer_config_dict = {k: v for k, v in optimizer_config_dict.items() if v is not None}
                if new_phase.get('learning_rate') is not None:
                    optimizer_config_dict['learning_rate'] = new_phase['learning_rate']
                optimizer = create_optimizer(model, optimizer_config_dict)

                if new_mode in _curriculum_head_cache:
                    # Restore optimizer state (momentum buffers, etc.)
                    # The param groups have changed (new head params), so we
                    # restore only backbone param states and let the head
                    # params pick up their cached momentum via param index mapping.
                    try:
                        optimizer.load_state_dict(_curriculum_head_cache[new_mode]['optimizer'])
                        print(f"[CURRICULUM] Restored cached optimizer state for '{new_mode}' mode")
                    except (ValueError, KeyError) as e:
                        print(f"[CURRICULUM] Could not restore optimizer state ({e}), using fresh optimizer")

                scheduler = CosineAnnealingLR(optimizer, T_max=new_phase['end_epoch'] - new_phase['start_epoch'] + 1, eta_min=1e-6)

                mode_label = f"superclass ({num_superclasses} classes)" if new_mode == 'superclass' else f"fine-class ({num_classes} classes)"
                cached = " [restored]" if new_mode in _curriculum_head_cache else " [fresh]"
                print(f"[CURRICULUM] Switched to {mode_label}{cached}, "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}")

                curriculum_current_phase = new_phase

        # Log epoch start to batch log file
        try:
            with open(batch_log_file, 'a') as f:
                f.write(json.dumps({
                    'type': 'epoch_start',
                    'epoch': epoch,
                    'total_epochs': total_epochs,
                    'elapsed_time': time.time() - start_time,
                }) + '\n')
        except Exception:
            pass

        # Progress callback for intra-epoch updates and accuracy monitoring
        def batch_progress_callback(batch_idx, total_batches, current_loss, current_acc):
            nonlocal milestone_index, threshold_snapshot_taken, epoch_snapshot_taken, total_weight_updates, latest_val_accuracy, latest_val_loss, milestone_wu_reached

            # Check for cancellation during training
            check_cancellation()

            # Build base progress message
            progress_msg = {
                'status': 'training',
                'current_epoch': epoch,
                'batches_complete': batch_idx,
                'total_batches': total_batches,
                'batch_loss': current_loss,
                'batch_accuracy': current_acc,
                'latest_val_accuracy': latest_val_accuracy,
                'latest_val_loss': latest_val_loss,
            }

            # Log batch-level metrics to detailed log file
            batch_log_entry = {
                'epoch': epoch,
                'batch': batch_idx,
                'total_batches': total_batches,
                'batch_progress': batch_idx / total_batches,
                'loss': current_loss,
                'accuracy': current_acc,
                'weight_updates': total_weight_updates + batch_idx,
            }

            # Extract mean activations periodically during training (check policy)
            include_activations = True
            if hasattr(netinit_config, 'logging_policy'):
                include_activations = netinit_config.logging_policy.batch_log_include_activations

            if include_activations and should_extract_activations(batch_idx, total_batches):
                # Check for cancellation before starting activation extraction
                check_cancellation()

                try:
                    print(f"  [BATCH {batch_idx}/{total_batches}] Extracting mean activations from validation set...")
                    # Switch to eval mode temporarily
                    was_training = model.training
                    model.eval()

                    # Create a small subset of validation data for fast activation extraction
                    # Take first few batches to get a quick snapshot
                    val_subset_batches = min(10, len(val_loader))  # Use first 10 batches
                    subset_activations = {i: [] for i in range(num_classes)}

                    # Register hook ONCE before the loop
                    activations = {}

                    def get_activation(name):
                        def hook(module, input, output):
                            activations[name] = output.detach()
                        return hook

                    # Register hook (same logic as extract_activations)
                    hook = None
                    if getattr(model, '_is_cct', False):  # CCT
                        hook = model.norm.register_forward_hook(get_activation('features'))
                    elif hasattr(model, 'fc'):  # ResNet
                        hook = model.avgpool.register_forward_hook(get_activation('features'))
                    elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):  # VGG
                        hook = model.features.register_forward_hook(get_activation('features'))
                    elif hasattr(model, 'heads'):  # ViT
                        hook = model.encoder.ln.register_forward_hook(get_activation('features'))
                    else:  # EfficientNet or others
                        for name, module in model.named_modules():
                            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
                                hook = module.register_forward_hook(get_activation('features'))
                                break

                    # Batch loop with pre-registered hook
                    with torch.no_grad():
                        for batch_count, (inputs, labels) in enumerate(val_loader):
                            if batch_count >= val_subset_batches:
                                break

                            inputs = inputs.to(device)
                            labels = labels.to(device)

                            # Forward pass - hook will capture activations
                            _ = model(inputs)

                            # Collect activations by class
                            if 'features' in activations:
                                batch_activations = activations['features']
                                if len(batch_activations.shape) == 3:
                                    # Transformer: [B, tokens, hidden]
                                    if getattr(model, '_activation_pool', 'cls') == 'mean':
                                        batch_activations = batch_activations.mean(dim=1)
                                    else:
                                        batch_activations = batch_activations[:, 0, :].contiguous()
                                elif len(batch_activations.shape) == 4:
                                    # CNN: [B, C, H, W] → global avg pool
                                    batch_activations = batch_activations.mean(dim=(2, 3))
                                elif len(batch_activations.shape) > 2:
                                    batch_activations = batch_activations.reshape(batch_activations.size(0), -1)

                                for i, label in enumerate(labels):
                                    class_idx = label.item()
                                    subset_activations[class_idx].append(batch_activations[i].cpu())

                            # Clear activations dict for next batch
                            activations.clear()

                    # Remove hook ONCE after loop
                    if hook:
                        hook.remove()

                    # Compute mean activations per class and their statistics
                    activation_stats = {}
                    for class_idx in range(num_classes):
                        if subset_activations[class_idx]:
                            class_acts = torch.stack(subset_activations[class_idx])
                            mean_act = class_acts.mean(dim=0)
                            activation_stats[f'class_{class_idx}_mean'] = float(mean_act.mean().item())
                            activation_stats[f'class_{class_idx}_std'] = float(mean_act.std().item())
                            activation_stats[f'class_{class_idx}_norm'] = float(torch.norm(mean_act).item())
                            activation_stats[f'class_{class_idx}_samples'] = len(subset_activations[class_idx])

                    batch_log_entry['activation_stats'] = activation_stats

                    # Restore training mode
                    if was_training:
                        model.train()

                    # Format activation norms for logging
                    norm_values = [activation_stats.get(f'class_{i}_norm', 0) for i in range(num_classes)]
                    norm_str = [f'{v:.3f}' for v in norm_values]
                    print(f"  [BATCH {batch_idx}/{total_batches}] Activation norms: {norm_str}")
                except Exception as e:
                    print(f"  [WARNING] Failed to extract activations at batch {batch_idx}: {e}")
                    batch_log_entry['activation_extraction_error'] = str(e)
                    # Ensure model is back in training mode
                    model.train()

            # Write batch log entry (append to JSONL file) - respect logging frequency
            should_log_batch = True
            if hasattr(netinit_config, 'logging_policy'):
                log_frequency = netinit_config.logging_policy.batch_log_frequency
                # Always log the last batch of each epoch
                should_log_batch = (batch_idx % log_frequency == 0) or (batch_idx == total_batches)

            if should_log_batch:
                try:
                    with open(batch_log_file, 'a') as f:
                        f.write(json.dumps(batch_log_entry) + '\n')
                except (FileNotFoundError, PermissionError, OSError):
                    # File may have been deleted by reset - check for cancellation
                    check_cancellation()
                except Exception as e:
                    print(f"  [WARNING] Failed to write batch log: {e}")

            # Weight-update milestone check (every batch, no validation needed to trigger)
            if intermediate_milestone_weight_updates and not epoch_snapshot_taken:
                current_wu = total_weight_updates + batch_idx
                for i, target_wu in enumerate(intermediate_milestone_weight_updates):
                    if not milestone_wu_reached[i]:
                        if current_wu >= target_wu:
                            print(f"\n[MILESTONE] Weight-update milestone {i+1}: TARGET={target_wu} ACTUAL={current_wu} at epoch {epoch}, batch {batch_idx}")

                            # Save training mode
                            was_training_wu = model.training

                            # Run validation to record accuracy in metadata
                            check_cancellation()
                            wu_val_loss, wu_val_acc = validate(model, val_loader, criterion, device)
                            wu_train_loss, wu_train_acc = validate(model, train_loader, criterion, device)
                            _clear_memory_cache()

                            elapsed_time = time.time() - start_time

                            write_progress({
                                'status': 'milestone_reached',
                                'current_epoch': epoch,
                                'batch_idx': batch_idx,
                                'milestone_index': milestone_index,
                                'target_weight_updates': target_wu,
                                'actual_weight_updates': current_wu,
                            })

                            snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"
                            snapshot_info = capture_snapshot(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=device,
                                snapshot_dir=snapshot_dir,
                                netinit_config=netinit_config,
                                num_classes=num_classes,
                                milestone_index=milestone_index,
                                epoch=epoch,
                                train_acc=wu_train_acc,
                                val_acc=wu_val_acc,
                                train_loss=wu_train_loss,
                                val_loss=wu_val_loss,
                                total_weight_updates=current_wu,
                                elapsed_time=elapsed_time,
                                snapshot_type='intermediate',
                                write_progress_fn=write_progress,
                                check_cancellation_fn=check_cancellation,
                                metrics_dir=Path(metrics_dir),
                                snapshots_info_list=snapshots_info,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                curriculum_swap=_get_curriculum_swap(),
                            )
                            snapshots_info.append(snapshot_info)
                            _clear_memory_cache()

                            milestone_wu_reached[i] = True
                            milestone_index += 1
                            epoch_snapshot_taken = True

                            # Update latest val metrics from the validation we just ran
                            latest_val_accuracy = wu_val_acc
                            latest_val_loss = wu_val_loss

                            write_progress({
                                'status': 'training',
                                'snapshot_captured': snapshot_info,
                                'current_epoch': epoch,
                                'current_milestone': milestone_index,
                            })

                            # Restore training mode
                            if was_training_wu:
                                model.train()

                            # One milestone per batch check
                            break
                        else:
                            # WU milestones are sorted — if this one isn't reached, later ones won't be
                            break

            # Intra-epoch validation for timely snapshot capture
            # Run validation checks even after snapshots to provide continuous progress updates
            if should_check_intra_epoch(batch_idx, total_batches):
                # Check for cancellation before starting potentially long validation
                check_cancellation()

                # Save training mode - validate() sets model.eval() but we must
                # restore model.train() so BatchNorm continues updating running stats
                was_training = model.training

                # Quick validation check with configurable set size
                if hasattr(netinit_config, 'validation_policy'):
                    val_fraction = netinit_config.validation_policy.validation_set_fraction
                    if val_fraction < 1.0:
                        # Create subset of validation data
                        subset_size = max(1, int(len(val_loader.dataset) * val_fraction))
                        subset_indices = list(range(subset_size))
                        val_subset = Subset(val_loader.dataset, subset_indices)
                        val_subset_loader = DataLoader(
                            val_subset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=dl_workers
                        )
                        intra_val_loss, intra_val_acc = validate(model, val_subset_loader, criterion, device)
                    else:
                        intra_val_loss, intra_val_acc = validate(model, val_loader, criterion, device)
                else:
                    # Backward compatibility: use full validation set
                    intra_val_loss, intra_val_acc = validate(model, val_loader, criterion, device)

                # Clear GPU cache after validation
                _clear_memory_cache()

                elapsed_time = time.time() - start_time

                # Update latest validation metrics
                latest_val_accuracy = intra_val_acc
                latest_val_loss = intra_val_loss

                print(f"  [INTRA-EPOCH] Batch {batch_idx}/{total_batches} - Val Acc: {intra_val_acc:.2f}%")

                # Add intra-epoch validation results to progress message
                progress_msg['intra_val_accuracy'] = intra_val_acc
                progress_msg['intra_val_loss'] = intra_val_loss
                progress_msg['latest_val_accuracy'] = latest_val_accuracy
                progress_msg['latest_val_loss'] = latest_val_loss

                # Check for intermediate accuracy milestones
                # Capture ALL reached milestones in sequence to avoid skipping
                for i, target_acc in enumerate(intermediate_milestone_accuracies):
                    if not milestone_accuracy_reached[i]:
                        # Check if this milestone has been reached
                        if intra_val_acc >= target_acc:
                            # This milestone has been reached for the first time
                            print(f"\n[MILESTONE] Reached intermediate accuracy milestone {i+1}: TARGET={target_acc:.2f}% ACTUAL={intra_val_acc:.2f}% at epoch {epoch}, batch {batch_idx}")

                            # Broadcast milestone reached
                            write_progress({
                                'status': 'milestone_reached',
                                'current_epoch': epoch,
                                'batch_idx': batch_idx,
                                'milestone_index': milestone_index,
                                'target_accuracy': target_acc,
                                'actual_accuracy': intra_val_acc,
                            })

                            snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"

                            # Check cancellation before validation
                            check_cancellation()

                            # Get current train metrics
                            train_val_loss, train_val_acc = validate(model, train_loader, criterion, device)

                            snapshot_info = capture_snapshot(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                device=device,
                                snapshot_dir=snapshot_dir,
                                netinit_config=netinit_config,
                                num_classes=num_classes,
                                milestone_index=milestone_index,
                                epoch=epoch,
                                train_acc=train_val_acc,
                                val_acc=intra_val_acc,
                                train_loss=train_val_loss,
                                val_loss=intra_val_loss,
                                total_weight_updates=total_weight_updates + batch_idx,
                                elapsed_time=elapsed_time,
                                snapshot_type='intermediate',
                                write_progress_fn=write_progress,
                                check_cancellation_fn=check_cancellation,
                                metrics_dir=Path(metrics_dir),
                                snapshots_info_list=snapshots_info,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                curriculum_swap=_get_curriculum_swap(),
                            )
                            snapshots_info.append(snapshot_info)

                            # Clear GPU cache after snapshot
                            _clear_memory_cache()

                            milestone_accuracy_reached[i] = True
                            milestone_index += 1
                            epoch_snapshot_taken = True

                            write_progress({
                                'status': 'training',
                                'snapshot_captured': snapshot_info,
                                'current_epoch': epoch,
                                'current_milestone': milestone_index,  # Next milestone to capture (already incremented above)
                            })
                            # Capture only ONE milestone per intra-epoch check to avoid excessive snapshots
                            break
                        else:
                            # Stop checking - if this milestone isn't reached yet, later ones won't be either
                            break

                # Check for threshold-based terminal snapshot during epoch
                if (accuracy_threshold and intra_val_acc >= accuracy_threshold and
                    not threshold_snapshot_taken and terminal_capture in ['at_threshold', 'both'] and
                    not epoch_snapshot_taken):

                    # Log warning if intermediate milestones were skipped
                    all_intermediate_milestones_reached = all(milestone_accuracy_reached)
                    if not all_intermediate_milestones_reached:
                        skipped = [i for i, reached in enumerate(milestone_accuracy_reached) if not reached]
                        print(f"\n[WARNING] Terminal threshold reached but {len(skipped)} intermediate milestone(s) were skipped: {skipped}")

                    print(f"\n[TERMINAL] Threshold {accuracy_threshold:.2f}% reached at epoch {epoch}, batch {batch_idx}")
                    snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"

                    # Check cancellation before validation
                    check_cancellation()

                    # Get current train metrics
                    train_val_loss, train_val_acc = validate(model, train_loader, criterion, device)

                    snapshot_info = capture_snapshot(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        snapshot_dir=snapshot_dir,
                        netinit_config=netinit_config,
                        num_classes=num_classes,
                        milestone_index=milestone_index,
                        epoch=epoch,
                        train_acc=train_val_acc,
                        val_acc=intra_val_acc,
                        train_loss=train_val_loss,
                        val_loss=intra_val_loss,
                        total_weight_updates=total_weight_updates + batch_idx,
                        elapsed_time=elapsed_time,
                        snapshot_type='threshold',
                        write_progress_fn=write_progress,
                        check_cancellation_fn=check_cancellation,
                        metrics_dir=Path(metrics_dir),
                        snapshots_info_list=snapshots_info,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        curriculum_swap=_get_curriculum_swap(),
                    )
                    snapshots_info.append(snapshot_info)

                    # Clear GPU cache after snapshot
                    _clear_memory_cache()

                    milestone_index += 1
                    threshold_snapshot_taken = True
                    epoch_snapshot_taken = True

                    write_progress({
                        'status': 'training',
                        'snapshot_captured': snapshot_info,
                        'current_epoch': epoch,
                    })

                # Restore training mode after all validation/snapshot operations
                if was_training:
                    model.train()

            # Write consolidated progress message (includes batch info and intra-epoch validation if it ran)
            write_progress(progress_msg)

        # Train epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            progress_callback=batch_progress_callback
        )

        # Check for cancellation before end-of-epoch validation
        check_cancellation()

        # Validate at end of epoch
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Clear GPU cache after validation
        _clear_memory_cache()

        # Update latest validation metrics
        latest_val_accuracy = val_acc
        latest_val_loss = val_loss

        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        total_weight_updates += len(train_loader)

        # Save metrics
        epoch_metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr'],
        }
        metrics_history.append(epoch_metrics)

        # Flush metrics periodically if configured
        if hasattr(netinit_config, 'logging_policy'):
            flush_interval = netinit_config.logging_policy.metrics_flush_interval
            keep_recent = netinit_config.logging_policy.metrics_keep_recent

            if epoch % flush_interval == 0:
                metrics_history = flush_metrics_to_disk(
                    metrics_history,
                    Path(metrics_dir) / "devtrain_metrics.json",
                    keep_recent=keep_recent
                )

        # Write progress with latest metrics
        write_progress({
            'status': 'training',
            'current_epoch': epoch,
            'latest_metrics': epoch_metrics,
            'total_weight_updates': total_weight_updates,
            'elapsed_time_seconds': elapsed_time,
        })

        print(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Time: {epoch_time:.1f}s")

        scheduler.step()

        # Log epoch completion to batch log file
        try:
            with open(batch_log_file, 'a') as f:
                f.write(json.dumps({
                    'type': 'epoch_complete',
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'epoch_time': epoch_time,
                    'elapsed_time': elapsed_time,
                    'weight_updates': total_weight_updates,
                }) + '\n')
        except Exception:
            pass

        # Check for weight-update milestone at epoch boundary (catch milestones that
        # land exactly on the epoch-end weight-update count)
        if not epoch_snapshot_taken and intermediate_milestone_weight_updates:
            for i, target_wu in enumerate(intermediate_milestone_weight_updates):
                if not milestone_wu_reached[i]:
                    if total_weight_updates >= target_wu:
                        print(f"\n[MILESTONE] Weight-update milestone {i+1} (end-of-epoch): TARGET={target_wu} ACTUAL={total_weight_updates}")

                        write_progress({
                            'status': 'milestone_reached',
                            'current_epoch': epoch,
                            'milestone_index': milestone_index,
                            'target_weight_updates': target_wu,
                            'actual_weight_updates': total_weight_updates,
                        })

                        snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"
                        snapshot_info = capture_snapshot(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device,
                            snapshot_dir=snapshot_dir,
                            netinit_config=netinit_config,
                            num_classes=num_classes,
                            milestone_index=milestone_index,
                            epoch=epoch,
                            train_acc=train_acc,
                            val_acc=val_acc,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            total_weight_updates=total_weight_updates,
                            elapsed_time=elapsed_time,
                            snapshot_type='intermediate',
                            write_progress_fn=write_progress,
                            check_cancellation_fn=check_cancellation,
                            metrics_dir=Path(metrics_dir),
                            snapshots_info_list=snapshots_info,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            curriculum_swap=_get_curriculum_swap(),
                        )
                        snapshots_info.append(snapshot_info)
                        _clear_memory_cache()

                        milestone_wu_reached[i] = True
                        milestone_index += 1

                        write_progress({
                            'status': 'training',
                            'snapshot_captured': snapshot_info,
                            'current_epoch': epoch,
                        })
                        break
                    else:
                        break

        # Check for accuracy-based intermediate milestone snapshots (only if not already taken intra-epoch)
        if not epoch_snapshot_taken:
            # Check all milestones to see if any have been newly reached
            # Iterate in order and stop at first un-reached milestone
            for i, target_acc in enumerate(intermediate_milestone_accuracies):
                if not milestone_accuracy_reached[i]:
                    # Check if this milestone has been reached
                    if val_acc >= target_acc:
                        # This milestone has been reached for the first time
                        print(f"\n[MILESTONE] Reached intermediate accuracy milestone {i+1}: TARGET={target_acc:.2f}% ACTUAL={val_acc:.2f}%")

                        # Broadcast milestone reached
                        write_progress({
                            'status': 'milestone_reached',
                            'current_epoch': epoch,
                            'milestone_index': milestone_index,
                            'target_accuracy': target_acc,
                            'actual_accuracy': val_acc,
                        })

                        snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"
                        snapshot_info = capture_snapshot(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            device=device,
                            snapshot_dir=snapshot_dir,
                            netinit_config=netinit_config,
                            num_classes=num_classes,
                            milestone_index=milestone_index,
                            epoch=epoch,
                            train_acc=train_acc,
                            val_acc=val_acc,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            total_weight_updates=total_weight_updates,
                            elapsed_time=elapsed_time,
                            snapshot_type='intermediate',
                            write_progress_fn=write_progress,
                            check_cancellation_fn=check_cancellation,
                            metrics_dir=Path(metrics_dir),
                            snapshots_info_list=snapshots_info,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            curriculum_swap=_get_curriculum_swap(),
                        )
                        snapshots_info.append(snapshot_info)

                        # Clear GPU cache after snapshot
                        _clear_memory_cache()

                        milestone_accuracy_reached[i] = True
                        milestone_index += 1

                        # Broadcast snapshot captured
                        write_progress({
                            'status': 'training',
                            'snapshot_captured': snapshot_info,
                            'current_epoch': epoch,
                        })
                        # Capture only ONE milestone per epoch check to avoid excessive snapshots
                        break
                    else:
                        # Stop checking - if this milestone isn't reached yet, later ones won't be either
                        break

        # Check for threshold-based terminal snapshot (only if not already taken intra-epoch)
        if (accuracy_threshold and val_acc >= accuracy_threshold and
            not threshold_snapshot_taken and not epoch_snapshot_taken):
            if terminal_capture in ['at_threshold', 'both']:
                # Log warning if intermediate milestones were skipped
                all_intermediate_milestones_reached = all(milestone_accuracy_reached)
                if not all_intermediate_milestones_reached:
                    skipped = [i for i, reached in enumerate(milestone_accuracy_reached) if not reached]
                    print(f"\n[WARNING] Terminal threshold reached but {len(skipped)} intermediate milestone(s) were skipped: {skipped}")

                print(f"\n[TERMINAL] Reached accuracy threshold {accuracy_threshold:.2f}% at epoch {epoch}")
                snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"
                snapshot_info = capture_snapshot(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    snapshot_dir=snapshot_dir,
                    netinit_config=netinit_config,
                    num_classes=num_classes,
                    milestone_index=milestone_index,
                    epoch=epoch,
                    train_acc=train_acc,
                    val_acc=val_acc,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    total_weight_updates=total_weight_updates,
                    elapsed_time=elapsed_time,
                    snapshot_type='threshold',
                    write_progress_fn=write_progress,
                    check_cancellation_fn=check_cancellation,
                    metrics_dir=Path(metrics_dir),
                    snapshots_info_list=snapshots_info,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    curriculum_swap=_get_curriculum_swap(),
                )
                snapshots_info.append(snapshot_info)

                # Clear GPU cache after snapshot
                _clear_memory_cache()

                milestone_index += 1
                threshold_snapshot_taken = True

                # Broadcast snapshot captured
                write_progress({
                    'status': 'training',
                    'snapshot_captured': snapshot_info,
                    'current_epoch': epoch,
                })

                # Stop training if terminal_capture is 'at_threshold' or 'both'
                # 'both' means: stop at whichever comes first (threshold OR final epoch)
                if terminal_capture in ['at_threshold', 'both']:
                    print(f"[TERMINAL] Stopping training after threshold snapshot (terminal_capture={terminal_capture})")
                    break

        # Check if we should stop due to accuracy threshold
        # Note: 'at_threshold' and 'both' will break above after taking threshold snapshot
        # 'final_epoch' will continue to the final epoch even after threshold is reached
        if accuracy_threshold and val_acc >= accuracy_threshold:
            if terminal_capture in ['at_threshold', 'both']:
                # Already handled above - training stopped
                pass
            elif terminal_capture == 'final_epoch':
                # Continue to final epoch even after threshold
                pass

    # Training complete
    total_time = time.time() - start_time
    final_epoch = epoch

    # Determine if we need to capture a terminal snapshot
    # We should capture a terminal snapshot if:
    # 1. Policy is 'final_epoch' and we reached final epoch
    # 2. Policy is 'both' and we reached final epoch (without hitting threshold, or we wouldn't be here)
    # 3. Policy is 'at_threshold' or 'both' but threshold was never reached (need a terminal snapshot anyway)
    # 4. Any case where we don't have a terminal snapshot yet and reached the end
    need_terminal_snapshot = False
    terminal_type = 'final'

    if terminal_capture == 'final_epoch' and final_epoch == total_epochs:
        # Policy explicitly requires final epoch snapshot
        need_terminal_snapshot = True
        terminal_type = 'final'
    elif terminal_capture == 'both' and final_epoch == total_epochs and not threshold_snapshot_taken:
        # 'both' means stop at first condition met (threshold OR final epoch)
        # If we're here at final epoch and no threshold snapshot was taken, we need one now
        need_terminal_snapshot = True
        terminal_type = 'final'
    elif terminal_capture in ['at_threshold', 'both'] and not threshold_snapshot_taken:
        # Threshold policy but threshold never reached - capture terminal snapshot anyway
        need_terminal_snapshot = True
        terminal_type = 'final'
        print(f"\n[TERMINAL] Accuracy threshold not reached, capturing final snapshot at epoch {final_epoch}")
    elif not threshold_snapshot_taken and final_epoch == total_epochs:
        # Fallback: always ensure we have a terminal snapshot at the end of training
        need_terminal_snapshot = True
        terminal_type = 'final'
        print(f"\n[TERMINAL] Ensuring terminal snapshot at end of training (epoch {final_epoch})")

    if need_terminal_snapshot:
        print(f"\n[TERMINAL] Capturing {terminal_type} snapshot at epoch {final_epoch}")
        # Use the last computed metrics
        snapshot_dir = Path(snapshots_base_dir) / f"milestone_{milestone_index}"
        snapshot_info = capture_snapshot(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            snapshot_dir=snapshot_dir,
            netinit_config=netinit_config,
            num_classes=num_classes,
            milestone_index=milestone_index,
            epoch=final_epoch,
            train_acc=train_acc,
            val_acc=val_acc,
            train_loss=train_loss,
            val_loss=val_loss,
            total_weight_updates=total_weight_updates,
            elapsed_time=total_time,
            snapshot_type=terminal_type,
            write_progress_fn=write_progress,
            check_cancellation_fn=check_cancellation,
            metrics_dir=Path(metrics_dir),
            snapshots_info_list=snapshots_info,
            optimizer=optimizer,
            scheduler=scheduler,
            curriculum_swap=_get_curriculum_swap(),
        )
        snapshots_info.append(snapshot_info)

        # Clear GPU cache after terminal snapshot
        _clear_memory_cache()

        # Broadcast snapshot captured
        write_progress({
            'status': 'training',
            'snapshot_captured': snapshot_info,
            'current_epoch': final_epoch,
        })

    print(f"\n[COMPLETE] Training finished after {final_epoch} epochs in {total_time:.1f}s")

    # Save final files
    metrics_file = Path(metrics_dir) / "devtrain_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"[COMPLETE] Saved metrics to {metrics_file}")

    snapshots_file = Path(metrics_dir) / "devtrain_snapshots.json"
    with open(snapshots_file, 'w') as f:
        json.dump(snapshots_info, f, indent=2)
    print(f"[COMPLETE] Saved snapshot metadata to {snapshots_file}")

    # Write final progress file (terminal snapshot is the last milestone)
    if snapshots_info:
        final_snapshot = snapshots_info[-1]
        write_progress({
            'status': 'complete',
            'snapshot_captured': final_snapshot,
            'current_epoch': final_epoch,
        })
        print(f"[COMPLETE] Saved final progress file")

    # Compute final class metrics
    train_class_metrics = compute_class_metrics(model, train_loader, device, class_names)
    val_class_metrics = compute_class_metrics(model, val_loader, device, class_names)

    # Save class metrics
    train_class_file = Path(metrics_dir) / "devtrain_class_metrics_train.json"
    with open(train_class_file, 'w') as f:
        json.dump([m.model_dump() if hasattr(m, 'model_dump') else m for m in train_class_metrics], f, indent=2)

    val_class_file = Path(metrics_dir) / "devtrain_class_metrics.json"
    with open(val_class_file, 'w') as f:
        json.dump([m.model_dump() if hasattr(m, 'model_dump') else m for m in val_class_metrics], f, indent=2)

    # Return summary
    return {
        'total_snapshots': len(snapshots_info),
        'final_epoch': final_epoch,
        'final_train_accuracy': train_acc,
        'final_val_accuracy': val_acc,
        'final_train_loss': train_loss,
        'final_val_loss': val_loss,
        'total_training_time_seconds': total_time,
        'milestone_accuracies': intermediate_milestone_accuracies,
        'terminal_capture': terminal_capture,
        'snapshot_policy': snapshot_policy_info,
    }
