"""Network initialization (NETINIT) router: endpoints for configuring a NETINIT job."""
import logging
import platform
import psutil
import random
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status

from ..netinit_models import (
    NetworkArchitecture,
    InitializationOption,
    NetInitConfiguration,
    HardwareSpecs,
    HardwareRecommendations,
    TimeEstimate,
    OptimizerConfig,
    OptimizerType,
    NETWORK_ARCHITECTURES,
    INITIALIZATION_OPTIONS,
    DEFAULT_OPTIMIZER_CONFIGS,
)
from ..storage import ensure_lane_structure, get_initial_weights_path, get_pretrained_weights_path
from ..database import StoveDB, LaneDB, ExperimentDB, ExperimentDatasetDB, PretrainedWeightsDB
from ..download_manager import download_pretrained_weights

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/netinit", tags=["netinit"])


def _compute_recommendations(ram_gb: float, device_type: str) -> HardwareRecommendations:
    """Recommended settings for the available hardware.

    On MPS (Apple Silicon) memory is shared with the GPU; the ML budget is ~60% of total RAM.
    """
    effective_ram = ram_gb * 0.6 if device_type == "mps" else ram_gb

    if effective_ram < 8:
        return HardwareRecommendations(
            disk_based_linear_probes=True, disk_based_multilayer=True,
            linear_probe_batch_size=32, samples_per_class=100,
            milestone_count=5, validation_set_fraction=0.3, batch_size=32,
            rationale=f"Very limited RAM ({ram_gb:.0f} GB). All disk-based modes forced. Minimal snapshots."
        )
    elif effective_ram < 14:
        return HardwareRecommendations(
            disk_based_linear_probes=True, disk_based_multilayer=True,
            linear_probe_batch_size=64, samples_per_class=150,
            milestone_count=8, validation_set_fraction=0.5, batch_size=64,
            rationale=f"{'Apple Silicon ' if device_type == 'mps' else ''}~{effective_ram:.0f} GB effective RAM. Disk-based mode strongly recommended. samples_per_class capped at 150."
        )
    elif effective_ram < 20:
        return HardwareRecommendations(
            disk_based_linear_probes=True, disk_based_multilayer=True,
            linear_probe_batch_size=64, samples_per_class=300,
            milestone_count=10, validation_set_fraction=0.5, batch_size=64,
            rationale=f"{'Apple Silicon ' if device_type == 'mps' else ''}~{effective_ram:.0f} GB effective RAM. Disk-based mode recommended. samples_per_class=300."
        )
    elif effective_ram < 40:
        return HardwareRecommendations(
            disk_based_linear_probes=True, disk_based_multilayer=True,
            linear_probe_batch_size=128, samples_per_class=500,
            milestone_count=15, validation_set_fraction=1.0, batch_size=128,
            rationale=f"~{effective_ram:.0f} GB RAM. Disk-based mode recommended for safety. Full validation set OK."
        )
    else:
        return HardwareRecommendations(
            disk_based_linear_probes=False, disk_based_multilayer=False,
            linear_probe_batch_size=256, samples_per_class=None,
            milestone_count=20, validation_set_fraction=1.0, batch_size=256,
            rationale=f"~{effective_ram:.0f} GB RAM. All samples fit in RAM. Disk-based mode optional."
        )


@router.get("/network-architectures", response_model=List[NetworkArchitecture])
async def get_network_architectures():
    """Get list of available network architectures."""
    return NETWORK_ARCHITECTURES


@router.get("/initialization-options", response_model=List[InitializationOption])
async def get_initialization_options():
    """Get list of available weight initialization methods."""
    return INITIALIZATION_OPTIONS


@router.get("/optimizer-defaults/{optimizer_type}", response_model=OptimizerConfig)
async def get_optimizer_defaults(optimizer_type: OptimizerType):
    """Get default configuration for a specific optimizer."""
    if optimizer_type not in DEFAULT_OPTIMIZER_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown optimizer type: {optimizer_type}"
        )
    return DEFAULT_OPTIMIZER_CONFIGS[optimizer_type]


@router.get("/hardware-specs", response_model=HardwareSpecs)
async def get_hardware_specs():
    """Get system hardware specifications."""
    try:
        cpu_count = psutil.cpu_count(logical=False) or 1
        cpu_model = platform.processor() or "Unknown CPU"

        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024 ** 3)

        # Basic GPU detection
        gpu_available = False
        gpu_count = 0
        gpu_models = []
        gpu_memory_gb = []
        mps_available = False
        device_type = "cpu"

        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    gpu_models.append(torch.cuda.get_device_name(i))
                    gpu_memory_gb.append(
                        torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                    )
        except ImportError:
            logger.warning("PyTorch not available, cannot detect GPU")
        except Exception as e:
            logger.warning(f"Error detecting GPU: {e}")

        # MPS detection (Apple Silicon)
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_available = True
                if not gpu_available:  # MPS counts as accelerator
                    gpu_available = True
                    gpu_count = 1
                    gpu_models = ["Apple Silicon (MPS)"]
                    gpu_memory_gb = [total_memory_gb]  # unified memory
        except Exception as e:
            logger.warning(f"Error detecting MPS: {e}")

        try:
            import torch
            if torch.cuda.is_available():
                device_type = "cuda"
            elif mps_available:
                device_type = "mps"
            else:
                device_type = "cpu"
        except Exception:
            pass

        recommended_settings = _compute_recommendations(total_memory_gb, device_type)

        return HardwareSpecs(
            cpu_count=cpu_count,
            cpu_model=cpu_model,
            total_memory_gb=total_memory_gb,
            gpu_available=gpu_available,
            gpu_count=gpu_count,
            gpu_models=gpu_models,
            gpu_memory_gb=gpu_memory_gb,
            mps_available=mps_available,
            device_type=device_type,
            recommended_settings=recommended_settings,
        )
    except Exception as e:
        logger.error(f"Error getting hardware specs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get hardware specifications"
        )


@router.post("/estimate-time", response_model=TimeEstimate)
async def estimate_training_time(config: NetInitConfiguration):
    """Estimate training time and resource requirements for a configuration."""
    try:
        network_arch = next(
            (arch for arch in NETWORK_ARCHITECTURES if arch.id == config.network_type),
            None
        )
        if not network_arch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown network type: {config.network_type}"
            )

        # With both criteria set, the epoch count is used as an upper bound; with only an accuracy
        # threshold, a typical convergence time is assumed.
        if config.training_policy.epochs:
            total_epochs = config.training_policy.epochs
        elif config.training_policy.accuracy_threshold:
            # Rough estimate; actual convergence varies
            total_epochs = 150  # Conservative estimate
        else:
            # Not reachable after validation; fallback default
            total_epochs = 100

        total_snapshots = config.snapshot_policy.milestone_count

        # Terminal capture mode "both" adds one snapshot
        if config.snapshot_policy.terminal_capture == "both":
            total_snapshots += 1

        base_time_per_epoch = network_arch.typical_training_time_per_epoch_seconds

        # Base time per epoch from the architecture, scaled by device
        try:
            import torch
            if torch.cuda.is_available():
                time_per_epoch = base_time_per_epoch / 7.0  # CUDA ~7x faster than CPU
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                time_per_epoch = base_time_per_epoch / 4.0  # MPS ~4x faster than CPU
            else:
                time_per_epoch = base_time_per_epoch
        except ImportError:
            time_per_epoch = base_time_per_epoch

        training_time_seconds = time_per_epoch * total_epochs
        estimated_completion_hours = training_time_seconds / 3600.0

        # Snapshot size from the weight size (~4 bytes per parameter)
        model_size_gb = (network_arch.parameter_count * 4) / (1024 ** 3)
        # Optimizer state and metadata: ~2.5x the weights
        snapshot_size_gb = model_size_gb * 2.5
        storage_required_gb = snapshot_size_gb * total_snapshots

        return TimeEstimate(
            training_time_seconds=training_time_seconds,
            total_snapshots=total_snapshots,
            storage_required_gb=storage_required_gb,
            estimated_completion_hours=estimated_completion_hours,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error estimating training time: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to estimate training time"
        )


@router.post("/initialize/{stove_id}")
async def initialize_network_weights(stove_id: str):
    """Initialize network weights for a job.

    Loads the job configuration, builds the model, applies the initialization method,
    saves the initial weights and marks the job as completed.
    """
    try:
        import sys
        from pathlib import Path
        ml_package_path = Path(__file__).parent.parent.parent.parent / "ml" / "src"
        if str(ml_package_path) not in sys.path:
            sys.path.insert(0, str(ml_package_path))

        from models import create_network
        from initialization import initialize_weights, save_weights

        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        lane_id = stove['lane_id']
        config_data = stove['configuration']

        if not config_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job configuration not found"
            )

        # The configuration is already a dict on the database object
        config = NetInitConfiguration.model_validate(config_data)

        lane = await LaneDB.get_by_id(lane_id)
        if not lane:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lane {lane_id} not found"
            )
        experiment_id = lane['experiment_id']

        experiment = await ExperimentDB.get_by_id(experiment_id)
        if not experiment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment {experiment_id} not found"
            )

        selected_classes = experiment['selected_classes']
        num_classes = len(selected_classes)

        dataset_status = await ExperimentDatasetDB.get_by_experiment(experiment_id)
        if not dataset_status or dataset_status['download_status'] != 'completed':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must be downloaded before running NETINIT. Please download the dataset first."
            )

        seed = config.random_seed
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        pretrained = config.initialization_method == "pretrained_imagenet"
        pretrained_weights_path = None

        if pretrained:
            weights_status = await PretrainedWeightsDB.get_by_experiment_and_network(
                experiment_id, config.network_type
            )

            if not weights_status or weights_status['download_status'] != 'completed':
                logger.info(f"Downloading pretrained weights for {config.network_type}")
                await download_pretrained_weights(experiment_id, config.network_type)

            pretrained_weights_path = str(get_pretrained_weights_path(experiment_id, config.network_type))

        model = create_network(
            network_type=config.network_type,
            num_classes=num_classes,
            pretrained=pretrained,
            pretrained_weights_path=pretrained_weights_path,
        )

        # Skipped for pretrained weights
        if not pretrained:
            initialize_weights(model, config.initialization_method, seed=seed)

        ensure_lane_structure(experiment_id, lane_id)

        weights_path = get_initial_weights_path(experiment_id, lane_id)
        save_weights(model, str(weights_path))

        await StoveDB.update_status(stove_id, status="completed", progress=1.0)

        logger.info(f"Initialized network for job {stove_id}, saved to {weights_path}")

        return {
            "success": True,
            "stove_id": stove_id,
            "weights_path": str(weights_path),
            "seed_used": seed,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing network: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize network: {str(e)}"
        )
