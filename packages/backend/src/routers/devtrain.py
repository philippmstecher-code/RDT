"""Developmental training (DEVTRAIN) router: endpoints for a DEVTRAIN job."""
import logging
import asyncio
import json
import webbrowser
from typing import Optional, Literal
from pathlib import Path
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, status, BackgroundTasks
from fastapi.responses import FileResponse
import torch

from ..devtrain_models import (
    DevTrainStatus,
    DevTrainDashboard,
    StartDevTrainRequest,
    StartDevTrainResponse,
    DevTrainWebSocketMessage,
    SnapshotInfo,
    RepresentationEvolution,
    VisualizationItem,
    SnapshotVisualizations,
    VisualizationsResponse,
)
from ..termrep_models import TrainingMetrics, ClassInfo, TerminalSnapshot
from ..netinit_models import NetInitConfiguration
from ..storage import (
    ensure_lane_structure,
    get_initial_weights_path,
    get_dev_snapshots_base_dir,
    get_dev_snapshot_dir,
    get_dev_metrics_dir,
    get_metrics_dir,
)
from ..database import StoveDB, LaneDB, ExperimentDB
from ..datasets import get_dataset_by_id, get_class_thumbnail_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/devtrain", tags=["devtrain"])

# Per-job state: WebSocket connections, training tasks and visualization tasks
active_connections: dict[str, list[WebSocket]] = {}

active_training_tasks: dict[str, asyncio.Task] = {}

active_visualization_tasks: dict[str, list[asyncio.Task]] = {}


# Async I/O helpers (run blocking calls in the default executor)

async def async_read_json(file_path: Path) -> dict:
    """Read and parse a JSON file in the executor."""
    def _read():
        with open(file_path, 'r') as f:
            return json.load(f)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _read)


async def async_exists(path: Path) -> bool:
    """Path.exists in the executor."""
    return await asyncio.get_event_loop().run_in_executor(None, path.exists)


async def async_is_dir(path: Path) -> bool:
    """Path.is_dir in the executor."""
    return await asyncio.get_event_loop().run_in_executor(None, path.is_dir)


async def async_glob(path: Path, pattern: str) -> list[Path]:
    """Path.glob in the executor; returns a list of paths."""
    def _glob():
        return list(path.glob(pattern))
    return await asyncio.get_event_loop().run_in_executor(None, _glob)


async def async_load_torch(file_path: Path, map_location='cpu'):
    """torch.load in the executor.

    Loads the whole object into memory before returning.
    """
    def _load():
        return torch.load(file_path, map_location=map_location)
    return await asyncio.get_event_loop().run_in_executor(None, _load)


class ConnectionManager:
    """Manage WebSocket connections for live training updates"""

    @staticmethod
    async def connect(stove_id: str, websocket: WebSocket):
        """Connect a new WebSocket client"""
        await websocket.accept()
        if stove_id not in active_connections:
            active_connections[stove_id] = []
        active_connections[stove_id].append(websocket)
        logger.info(f"WebSocket connected for DEVTRAIN job {stove_id}")

    @staticmethod
    async def disconnect(stove_id: str, websocket: WebSocket):
        """Disconnect a WebSocket client"""
        if stove_id in active_connections:
            if websocket in active_connections[stove_id]:
                active_connections[stove_id].remove(websocket)
            if not active_connections[stove_id]:
                del active_connections[stove_id]
        logger.info(f"WebSocket disconnected for DEVTRAIN job {stove_id}")

    @staticmethod
    async def broadcast(stove_id: str, message: DevTrainWebSocketMessage):
        """Broadcast a message to all connected clients for a job"""
        if stove_id not in active_connections:
            return

        message_json = message.model_dump_json(exclude_none=True)
        disconnected = []

        for websocket in active_connections[stove_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(websocket)

        for websocket in disconnected:
            await ConnectionManager.disconnect(stove_id, websocket)


@router.websocket("/ws/{stove_id}")
async def websocket_endpoint(websocket: WebSocket, stove_id: str):
    """WebSocket endpoint for live training updates"""
    await ConnectionManager.connect(stove_id, websocket)
    try:
        while True:
            # The client sends nothing; the loop keeps the connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ConnectionManager.disconnect(stove_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error for DEVTRAIN job {stove_id}: {e}")
        await ConnectionManager.disconnect(stove_id, websocket)


async def validate_devtrain_dependencies(stove_id: str) -> tuple[dict, dict, dict, NetInitConfiguration]:
    """Check that NETINIT has completed for the lane.

    Returns (stove, lane, experiment, netinit_config); raises HTTPException otherwise.
    """
    stove = await StoveDB.get_by_id(stove_id)
    if not stove:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {stove_id} not found"
        )

    if stove['stove_type'].upper() != 'DEVTRAIN':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {stove_id} is not a DEVTRAIN job"
        )

    lane_id = stove['lane_id']
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

    stoves_in_lane = await StoveDB.get_all(lane_id)

    netinit_stove = next((s for s in stoves_in_lane if s['stove_type'].upper() == 'NETINIT'), None)
    if not netinit_stove:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NETINIT job not found in lane"
        )
    if netinit_stove['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"NETINIT must be completed before starting DEVTRAIN (current status: {netinit_stove['status']})"
        )
    if not netinit_stove['configuration']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NETINIT configuration not found"
        )

    netinit_config = NetInitConfiguration.model_validate(netinit_stove['configuration'])

    return stove, lane, experiment, netinit_config


@router.get("/status/{stove_id}", response_model=DevTrainStatus)
async def get_devtrain_status(stove_id: str):
    """Get current status of DEVTRAIN job"""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        if stove['stove_type'].upper() != 'DEVTRAIN':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {stove_id} is not a DEVTRAIN job"
            )

        lane_id = stove['lane_id']
        lane = await LaneDB.get_by_id(lane_id)
        if not lane:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lane {lane_id} not found"
            )

        stoves_in_lane = await StoveDB.get_all(lane_id)
        netinit_stove = next((s for s in stoves_in_lane if s['stove_type'].upper() == 'NETINIT'), None)

        total_epochs = None
        target_accuracy = None
        total_snapshots = 0

        if netinit_stove and netinit_stove['configuration']:
            config = NetInitConfiguration.model_validate(netinit_stove['configuration'])
            total_epochs = config.training_policy.epochs
            target_accuracy = config.training_policy.accuracy_threshold
            total_snapshots = config.snapshot_policy.milestone_count

        runtime_data = {}
        if stove['configuration']:
            runtime_data = stove['configuration']

        return DevTrainStatus(
            stove_id=stove_id,
            status=stove['status'],
            progress=stove['progress'],
            current_epoch=runtime_data.get('current_epoch'),
            total_epochs=total_epochs,
            current_accuracy=runtime_data.get('current_accuracy'),
            target_accuracy=target_accuracy,
            snapshots_captured=runtime_data.get('snapshots_captured', 0),
            total_snapshots=total_snapshots,
            current_milestone=runtime_data.get('current_milestone'),
            extracting_representations=runtime_data.get('extracting_representations', False),
            elapsed_time_seconds=runtime_data.get('elapsed_time_seconds'),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DEVTRAIN status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting status: {str(e)}"
        )


@router.get("/dashboard/{stove_id}", response_model=DevTrainDashboard)
async def get_devtrain_dashboard(stove_id: str):
    """Get complete dashboard data for DEVTRAIN job"""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        if stove['stove_type'].upper() != 'DEVTRAIN':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {stove_id} is not a DEVTRAIN job"
            )

        lane_id = stove['lane_id']
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

        metrics_dir = get_dev_metrics_dir(experiment_id, lane_id)

        # Metrics and snapshot files may be partially written while training runs
        metrics_history = []
        metrics_file = metrics_dir / "devtrain_metrics.json"
        if await async_exists(metrics_file):
            try:
                metrics_data = await async_read_json(metrics_file)
                metrics_history = [TrainingMetrics(**m) for m in metrics_data]
            except Exception:
                pass

        snapshots = []
        snapshots_file = metrics_dir / "devtrain_snapshots.json"
        if await async_exists(snapshots_file):
            try:
                snapshots_data = await async_read_json(snapshots_file)
                snapshots = [SnapshotInfo(**s) for s in snapshots_data]
            except Exception:
                pass

        dataset = get_dataset_by_id(experiment['dataset_id'])
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {experiment['dataset_id']} not found"
            )

        classes_info = []
        for class_id in experiment['selected_classes']:
            def find_class(nodes, target_id):
                for node in nodes:
                    if node.id == target_id:
                        return node
                    if node.children:
                        result = find_class(node.children, target_id)
                        if result:
                            return result
                return None

            class_node = find_class(dataset.hierarchy, class_id)
            if class_node:
                thumbnail_url = get_class_thumbnail_url(experiment['dataset_id'], class_id)
                classes_info.append(ClassInfo(
                    class_id=class_id,
                    class_name=class_node.name,
                    thumbnail_url=thumbnail_url
                ))

        error_message = None
        if stove['configuration'] and 'error_message' in stove['configuration']:
            error_message = stove['configuration']['error_message']

        terminal_snapshot = None

        # Milestone accuracies are read from a dedicated file because the progress file is
        # overwritten on every update
        effective_milestone_accuracies = None
        snapshot_policy_file = metrics_dir / "devtrain_snapshot_policy.json"
        if await async_exists(snapshot_policy_file):
            try:
                snapshot_policy = await async_read_json(snapshot_policy_file)
                if 'milestone_accuracies' in snapshot_policy:
                    effective_milestone_accuracies = snapshot_policy['milestone_accuracies']
            except Exception:
                pass

        visualizations_generated = False
        if snapshots:
            dev_snapshots_base_dir = get_dev_snapshots_base_dir(experiment_id, lane_id)
            for snapshot in snapshots:
                milestone_dir = dev_snapshots_base_dir / f"milestone_{snapshot.milestone_index}"
                vis_dir = milestone_dir / "visualizations"
                if await async_exists(vis_dir):
                    for method in ['deep_dream', 'gradient_saliency', 'feature_maps', 'projection']:
                        method_dir = vis_dir / method
                        if await async_exists(method_dir):
                            png_files = await async_glob(method_dir, "*.png")
                            if png_files:
                                visualizations_generated = True
                                break
                if visualizations_generated:
                    break

        return DevTrainDashboard(
            stove_id=stove_id,
            status=stove['status'],
            progress=stove['progress'],
            metrics_history=metrics_history,
            snapshots=snapshots,
            classes_info=classes_info,
            terminal_snapshot=terminal_snapshot,
            effective_milestone_accuracies=effective_milestone_accuracies,
            visualizations_generated=visualizations_generated,
            error_message=error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting DEVTRAIN dashboard: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting dashboard: {str(e)}"
        )


async def poll_and_broadcast_progress(stove_id: str, metrics_dir: Path, netinit_config: NetInitConfiguration):
    """Poll devtrain_progress.json and broadcast updates"""
    progress_file = metrics_dir / "devtrain_progress.json"
    last_epoch = 0
    last_batch = 0
    last_init_steps_count = 0  # number of init steps already broadcast
    last_snapshot_index = -1
    last_extraction_event = None  # (milestone_index, extraction_method)
    last_intra_val_event = None  # (epoch, batch, accuracy)
    last_milestone_reached = -1
    logger.info(f"Starting progress polling for DEVTRAIN job {stove_id}")

    while True:
        try:
            stove = await StoveDB.get_by_id(stove_id)
            if not stove or stove['status'] not in ['running', 'configuring']:
                logger.info(f"Stopping polling - job status: {stove['status'] if stove else 'not found'}")
                break

            # The progress file may be partially written while training runs
            if await async_exists(progress_file):
                try:
                    progress_data = await async_read_json(progress_file)
                except Exception:
                    await asyncio.sleep(0.5)
                    continue

                current_epoch = progress_data.get('current_epoch', 0)
                batches_complete = progress_data.get('batches_complete', 0)
                status_val = progress_data.get('status', 'training')

                # Init steps are accumulated by the trainer, so none are missed between polls
                init_steps = progress_data.get('initialization_steps', [])
                if len(init_steps) > last_init_steps_count:
                    for init_entry in init_steps[last_init_steps_count:]:
                        init_step = init_entry.get('step', '')
                        init_detail = init_entry.get('detail', '')
                        logger.info(f"Broadcasting initialization step: {init_step}")
                        await ConnectionManager.broadcast(
                            stove_id,
                            DevTrainWebSocketMessage(
                                message_type="initialization",
                                stove_id=stove_id,
                                initialization_step=init_step,
                                initialization_detail=init_detail,
                            )
                        )
                    last_init_steps_count = len(init_steps)

                if status_val == 'milestone_reached':
                    milestone_index = progress_data.get('milestone_index')
                    target_accuracy = progress_data.get('target_accuracy')
                    actual_accuracy = progress_data.get('actual_accuracy')
                    batch_idx = progress_data.get('batch_idx')

                    if (milestone_index is not None and target_accuracy is not None and
                        actual_accuracy is not None and milestone_index > last_milestone_reached):
                        logger.info(f"Broadcasting milestone reached: milestone {milestone_index}")
                        await ConnectionManager.broadcast(
                            stove_id,
                            DevTrainWebSocketMessage(
                                message_type="milestone_reached",
                                stove_id=stove_id,
                                milestone_index=milestone_index,
                                target_accuracy=target_accuracy,
                                actual_accuracy=actual_accuracy,
                                current_epoch=current_epoch,
                                batch_idx=batch_idx,
                            )
                        )
                        last_milestone_reached = milestone_index

                if 'intra_val_accuracy' in progress_data and 'intra_val_loss' in progress_data:
                    intra_val_acc = progress_data['intra_val_accuracy']
                    intra_val_loss = progress_data['intra_val_loss']
                    current_intra_val_event = (current_epoch, batches_complete, intra_val_acc)

                    if current_intra_val_event != last_intra_val_event:
                        logger.info(f"Broadcasting intra-validation: {intra_val_acc:.2f}%")
                        await ConnectionManager.broadcast(
                            stove_id,
                            DevTrainWebSocketMessage(
                                message_type="progress",
                                stove_id=stove_id,
                                current_epoch=current_epoch,
                                batches_complete=batches_complete,
                                total_batches=progress_data.get('total_batches', 1),
                                intra_val_accuracy=intra_val_acc,
                                intra_val_loss=intra_val_loss,
                            )
                        )
                        last_intra_val_event = current_intra_val_event

                if 'snapshot_captured' in progress_data:
                    snapshot_data = progress_data['snapshot_captured']
                    snapshot_index = snapshot_data.get('milestone_index', -1)
                    if snapshot_index > last_snapshot_index:
                        logger.info(f"Broadcasting snapshot captured: milestone {snapshot_index}")
                        await ConnectionManager.broadcast(
                            stove_id,
                            DevTrainWebSocketMessage(
                                message_type="snapshot_captured",
                                stove_id=stove_id,
                                snapshot_info=SnapshotInfo(**snapshot_data),
                            )
                        )
                        last_snapshot_index = snapshot_index

                        # current_milestone is the next milestone to capture
                        current_config = stove.get('configuration') or {}
                        current_milestone_from_progress = progress_data.get('current_milestone')
                        if current_milestone_from_progress is not None:
                            current_config['current_milestone'] = current_milestone_from_progress
                        await StoveDB.update_configuration(stove_id, current_config)

                if status_val == 'extracting_representations':
                    milestone_index = progress_data.get('milestone_index')
                    extraction_method = progress_data.get('extraction_method')
                    current_extraction_event = (milestone_index, extraction_method)

                    if milestone_index is not None and current_extraction_event != last_extraction_event:
                        logger.info(f"Broadcasting extracting representations: {extraction_method} for milestone {milestone_index}")
                        await ConnectionManager.broadcast(
                            stove_id,
                            DevTrainWebSocketMessage(
                                message_type="extracting_representations",
                                stove_id=stove_id,
                                milestone_index=milestone_index,
                                extraction_method=extraction_method,
                            )
                        )
                        last_extraction_event = current_extraction_event

                        current_config = stove.get('configuration') or {}
                        current_config['extracting_representations'] = True
                        current_config['current_milestone'] = milestone_index
                        await StoveDB.update_configuration(stove_id, current_config)

                if current_epoch > last_epoch and 'latest_metrics' in progress_data:
                    logger.info(f"Broadcasting epoch {current_epoch} for DEVTRAIN job {stove_id}")
                    metrics = TrainingMetrics(**progress_data['latest_metrics'])
                    await ConnectionManager.broadcast(
                        stove_id,
                        DevTrainWebSocketMessage(
                            message_type="epoch_complete",
                            stove_id=stove_id,
                            metrics=metrics,
                        )
                    )
                    last_epoch = current_epoch
                    last_batch = 0

                    current_config = stove.get('configuration') or {}
                    current_config.update({
                        'current_epoch': current_epoch,
                        'current_accuracy': progress_data['latest_metrics'].get('train_accuracy'),
                        'extracting_representations': False,
                    })
                    await StoveDB.update_configuration(stove_id, current_config)

                # Batch-level progress, throttled to the first batch, every 5 batches or every 5%
                if status_val == 'training' and batches_complete > last_batch and batches_complete > 0:
                    total_batches = progress_data.get('total_batches', 1)
                    batches_since_last = batches_complete - last_batch

                    progress_now = batches_complete / total_batches if total_batches > 0 else 0
                    progress_last = last_batch / total_batches if total_batches > 0 else 0
                    progress_milestone = int(progress_now * 20) > int(progress_last * 20)  # every 5%

                    should_broadcast = (
                        batches_complete == 1 or
                        batches_since_last >= 5 or
                        progress_milestone
                    )

                    if should_broadcast:
                        batch_loss = progress_data.get('batch_loss')
                        batch_accuracy = progress_data.get('batch_accuracy')
                        latest_val_accuracy = progress_data.get('latest_val_accuracy')
                        latest_val_loss = progress_data.get('latest_val_loss')

                        logger.info(f"Broadcasting batch progress: epoch {current_epoch}, batch {batches_complete}/{total_batches}, val_acc: {latest_val_accuracy:.2f}%" if latest_val_accuracy else f"Broadcasting batch progress: epoch {current_epoch}, batch {batches_complete}/{total_batches}")
                        await ConnectionManager.broadcast(
                            stove_id,
                            DevTrainWebSocketMessage(
                                message_type="progress",
                                stove_id=stove_id,
                                progress=progress_now,
                                current_epoch=current_epoch,
                                batches_complete=batches_complete,
                                total_batches=total_batches,
                                batch_loss=batch_loss,
                                batch_accuracy=batch_accuracy,
                                latest_val_accuracy=latest_val_accuracy,
                                latest_val_loss=latest_val_loss,
                            )
                        )

                    last_batch = batches_complete

                # Database progress from epoch and batch completion

                total_epochs = netinit_config.training_policy.epochs or 100

                epoch_progress = 0.0
                if total_epochs > 0:
                    epoch_progress = current_epoch / total_epochs

                    if current_epoch < total_epochs and batches_complete > 0:
                        total_batches_val = progress_data.get('total_batches', 1)
                        if total_batches_val > 0:
                            batch_fraction = batches_complete / total_batches_val
                            epoch_progress += (batch_fraction / total_epochs)

                progress = min(epoch_progress, 0.99)  # capped at 99% until complete

                await StoveDB.update_status(
                    stove_id,
                    status="running",
                    progress=progress,
                )

                # Keep the configuration in sync so the status endpoint returns the current epoch
                current_config = stove.get('configuration') or {}
                if current_config.get('current_epoch') != current_epoch:
                    current_config['current_epoch'] = current_epoch
                    await StoveDB.update_configuration(stove_id, current_config)

            await asyncio.sleep(0.5)

        except Exception as e:
            logger.error(f"Error polling progress: {e}")
            await asyncio.sleep(2)


async def run_devtrain_background(stove_id: str):
    """Background task: run DEVTRAIN training and broadcast updates over WebSocket."""
    try:
        import sys
        from pathlib import Path
        ml_package_path = Path(__file__).parent.parent.parent.parent / "ml" / "src"
        if str(ml_package_path) not in sys.path:
            sys.path.insert(0, str(ml_package_path))

        import importlib
        import devtrain as _devtrain_mod
        importlib.reload(_devtrain_mod)
        from devtrain import train_with_snapshots

        stove, lane, experiment, netinit_config = await validate_devtrain_dependencies(stove_id)

        experiment_id = experiment['id']
        lane_id = lane['id']

        ensure_lane_structure(experiment_id, lane_id)

        initial_weights_path = get_initial_weights_path(experiment_id, lane_id)
        snapshots_base_dir = get_dev_snapshots_base_dir(experiment_id, lane_id)
        metrics_dir = get_dev_metrics_dir(experiment_id, lane_id)

        snapshots_base_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        if not initial_weights_path.exists():
            raise Exception("Initial weights not found. Run NETINIT first.")

        # Resume interrupted training if snapshots exist
        resume_state = None
        snapshots_file = metrics_dir / "devtrain_snapshots.json"
        snapshot_policy_file = metrics_dir / "devtrain_snapshot_policy.json"
        progress_file = metrics_dir / "devtrain_progress.json"

        if stove['status'] == 'error' and snapshots_file.exists():
            try:
                import json as _json
                with open(snapshots_file) as f:
                    existing_snapshots = _json.load(f)

                existing_milestone_count = len(existing_snapshots)
                next_milestone_dir = snapshots_base_dir / f"milestone_{existing_milestone_count}"
                if next_milestone_dir.exists() and (next_milestone_dir / "checkpoint.pt").exists():
                    # Captured but not recorded in snapshots.json (interrupted during the write)
                    logger.info(f"Found unrecorded milestone_{existing_milestone_count} on disk, skipping it (incomplete)")

                if existing_snapshots and len(existing_snapshots) > 1:
                    # T=0 plus at least one intermediate snapshot
                    latest_snapshot = existing_snapshots[-1]
                    latest_checkpoint = Path(latest_snapshot['snapshot_dir']) / "checkpoint.pt"

                    if latest_checkpoint.exists():
                        milestone_accuracies = []
                        if snapshot_policy_file.exists():
                            with open(snapshot_policy_file) as f:
                                policy = _json.load(f)
                                milestone_accuracies = policy.get('milestone_accuracies', [])

                        # Snapshots after index 0 correspond to intermediate milestones
                        intermediate_snapshots = [s for s in existing_snapshots if not s.get('is_t0_baseline', False)]
                        milestones_reached = [False] * len(milestone_accuracies)
                        for i in range(min(len(intermediate_snapshots), len(milestones_reached))):
                            milestones_reached[i] = True

                        resume_epoch = latest_snapshot['epoch'] + 1
                        if progress_file.exists():
                            with open(progress_file) as f:
                                progress_data = _json.load(f)
                                if 'current_epoch' in progress_data:
                                    resume_epoch = progress_data['current_epoch']

                        total_weight_updates = latest_snapshot.get('weight_updates', 0)

                        resume_state = {
                            'checkpoint_path': str(latest_checkpoint),
                            'start_epoch': resume_epoch,
                            'existing_snapshots': existing_snapshots,
                            'next_milestone_index': existing_milestone_count,
                            'milestones_reached': milestones_reached,
                            'total_weight_updates': total_weight_updates,
                        }
                        logger.info(
                            f"Resuming DEVTRAIN from epoch {resume_epoch} with "
                            f"{len(existing_snapshots)} existing snapshots "
                            f"(checkpoint: {latest_checkpoint})"
                        )
            except Exception as e:
                logger.warning(f"Failed to build resume state, starting fresh: {e}")
                resume_state = None

        await StoveDB.update_status(stove_id, status="running", progress=0.0)

        await ConnectionManager.broadcast(
            stove_id,
            DevTrainWebSocketMessage(
                message_type="progress",
                stove_id=stove_id,
                progress=0.0,
                current_epoch=0,
            )
        )

        poll_task = asyncio.create_task(
            poll_and_broadcast_progress(stove_id, metrics_dir, netinit_config)
        )

        # Training runs in the executor
        import functools
        training_fn = functools.partial(
            train_with_snapshots,
            str(initial_weights_path),
            netinit_config,
            experiment['dataset_id'],
            experiment['selected_classes'],
            str(snapshots_base_dir),
            str(metrics_dir),
            experiment_id,
            stove_id,
            resume_state=resume_state,
        )
        summary = await asyncio.get_event_loop().run_in_executor(
            None,
            training_fn,
        )

        poll_task.cancel()
        try:
            await poll_task
        except asyncio.CancelledError:
            pass

        await StoveDB.update_status(stove_id, status="completed", progress=1.0)

        final_config = stove.get('configuration') or {}
        final_config['snapshots_captured'] = summary['total_snapshots']
        await StoveDB.update_configuration(stove_id, final_config)

        await ConnectionManager.broadcast(
            stove_id,
            DevTrainWebSocketMessage(
                message_type="training_complete",
                stove_id=stove_id,
            )
        )

        logger.info(f"DEVTRAIN completed for job {stove_id}")

        dashboard_url = f"http://localhost:5173/stoves/{stove_id}/devtrain"
        try:
            logger.info(f"Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            logger.warning(f"Failed to open browser automatically: {e}")

    except asyncio.CancelledError:
        # Cancelled, e.g. by a reset
        logger.info(f"DEVTRAIN cancelled for job {stove_id}")
        if 'poll_task' in locals():
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass
        raise

    except Exception as e:
        # Stop the polling task
        if 'poll_task' in locals():
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

        is_user_cancellation = "Training cancelled by user" in str(e)
        is_file_deleted = isinstance(e, (FileNotFoundError, PermissionError, OSError))

        # Files deleted during training (reset) count as cancellation; the reset already set the status
        if is_file_deleted:
            logger.info(f"DEVTRAIN stopped due to file deletion (reset) for job {stove_id}")
            return

        if is_user_cancellation:
            logger.info(f"DEVTRAIN stopped by user for job {stove_id}")
            return

        logger.error(f"Error running DEVTRAIN: {e}", exc_info=True)

        try:
            await StoveDB.update_status(stove_id, status="error", progress=0.0)

            stove = await StoveDB.get_by_id(stove_id)
            if stove:
                error_config = stove.get('configuration') or {}
                error_config['error_message'] = str(e)
                await StoveDB.update_configuration(stove_id, error_config)

            await ConnectionManager.broadcast(
                stove_id,
                DevTrainWebSocketMessage(
                    message_type="error",
                    stove_id=stove_id,
                    error_message=str(e),
                )
            )
        except Exception as update_error:
            logger.error(f"Error updating job after failure: {update_error}")

    finally:
        if stove_id in active_training_tasks:
            del active_training_tasks[stove_id]

        try:
            if 'metrics_dir' in locals():
                cancellation_file = metrics_dir / "devtrain_cancel.flag"
                if cancellation_file.exists():
                    cancellation_file.unlink()
                    logger.info(f"Cleaned up cancellation flag for job {stove_id}")
        except Exception as cleanup_error:
            logger.warning(f"Error cleaning up cancellation flag: {cleanup_error}")


@router.post("/start", response_model=StartDevTrainResponse)
async def start_devtrain(request: StartDevTrainRequest, background_tasks: BackgroundTasks):
    """Start DEVTRAIN training for a job"""
    try:
        stove_id = request.stove_id

        stove, lane, experiment, netinit_config = await validate_devtrain_dependencies(stove_id)

        if stove['status'] == 'running':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="DEVTRAIN is already running"
            )

        # Cancel running visualization tasks for this job
        if stove_id in active_visualization_tasks:
            tasks = active_visualization_tasks[stove_id]
            active_tasks = [t for t in tasks if not t.done()]
            if active_tasks:
                logger.info(f"Cancelling {len(active_tasks)} active visualization tasks before starting training for job {stove_id}")
                for task in active_tasks:
                    task.cancel()
                try:
                    await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=2.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"Error cancelling visualization tasks: {e}")
            active_visualization_tasks[stove_id] = []

        total_epochs = netinit_config.training_policy.epochs or 100
        # Rough estimate
        estimated_hours = (total_epochs * 60 * 1.2) / 3600  # 60 s per epoch plus 20% snapshot overhead

        task = asyncio.create_task(run_devtrain_background(stove_id))
        active_training_tasks[stove_id] = task

        return StartDevTrainResponse(
            success=True,
            stove_id=stove_id,
            message="DEVTRAIN training started",
            estimated_completion_hours=estimated_hours,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting DEVTRAIN: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting training: {str(e)}"
        )


def compute_evolution_metrics(snapshots_data: list, snapshots_base_dir: Path):
    """Evolution metrics across snapshots (blocking; run in the executor).

    Returns a dict of epochs and metric arrays: class separation from mean activations,
    best linear-probe accuracy and within-class variance.
    """
    import torch
    import torch.nn.functional as F

    epochs = [s['epoch'] for s in snapshots_data]
    class_separation_values = []
    probe_accuracy_values = []
    within_class_variance_values = []

    for snapshot_info in snapshots_data:
        milestone_idx = snapshot_info['milestone_index']
        snapshot_dir = snapshots_base_dir / f"milestone_{milestone_idx}"

        # Class separation: 1 - mean off-diagonal cosine similarity of class means
        mean_acts_file = snapshot_dir / "mean_activations.pt"
        if mean_acts_file.exists():
            try:
                mean_activations = torch.load(mean_acts_file, map_location='cpu')

                class_vectors = []
                for class_idx in sorted(mean_activations.keys()):
                    class_vectors.append(mean_activations[class_idx])

                if len(class_vectors) > 1:
                    class_matrix = torch.stack(class_vectors)
                    normalized = F.normalize(class_matrix, p=2, dim=1)
                    similarity_matrix = torch.mm(normalized, normalized.t())

                    num_classes = len(class_vectors)
                    off_diagonal_sum = similarity_matrix.sum() - num_classes
                    avg_similarity = off_diagonal_sum / (num_classes * (num_classes - 1))
                    avg_separation = 1.0 - avg_similarity.item()
                    class_separation_values.append(avg_separation)
                else:
                    class_separation_values.append(0.0)
            except Exception as e:
                logger.warning(f"Error computing class separation for milestone {milestone_idx}: {e}")
                class_separation_values.append(None)
        else:
            class_separation_values.append(None)

        # Best linear-probe validation accuracy across layers
        probes_file = snapshot_dir / "linear_probes.pt"
        if probes_file.exists():
            try:
                probe_results = torch.load(probes_file, map_location='cpu')
                best_val_acc = probe_results.get('best_val_accuracy', 0.0)
                probe_accuracy_values.append(best_val_acc)
            except Exception as e:
                logger.warning(f"Error loading probe accuracy for milestone {milestone_idx}: {e}")
                probe_accuracy_values.append(None)
        else:
            probe_accuracy_values.append(None)

        # Mean within-class variance from individual activations
        individual_acts_file = snapshot_dir / "individual_activations.pt"
        if individual_acts_file.exists():
            try:
                individual_data = torch.load(individual_acts_file, map_location='cpu')

                class_variances = []
                for class_idx in sorted(individual_data['individual'].keys()):
                    class_acts = individual_data['individual'][class_idx]
                    if len(class_acts) > 1:
                        variance = class_acts.var(dim=0).mean().item()
                        class_variances.append(variance)

                if class_variances:
                    avg_variance = sum(class_variances) / len(class_variances)
                    within_class_variance_values.append(avg_variance)
                else:
                    within_class_variance_values.append(None)
            except Exception as e:
                logger.warning(f"Error computing within-class variance for milestone {milestone_idx}: {e}")
                within_class_variance_values.append(None)
        else:
            within_class_variance_values.append(None)

    return {
        'epochs': epochs,
        'class_separation': class_separation_values,
        'probe_accuracy': probe_accuracy_values,
        'within_class_variance': within_class_variance_values,
    }


@router.get("/representation-evolution/{stove_id}", response_model=RepresentationEvolution)
async def get_representation_evolution(stove_id: str):
    """Get representation evolution data across all snapshots"""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        lane_id = stove['lane_id']
        lane = await LaneDB.get_by_id(lane_id)
        if not lane:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lane {lane_id} not found"
            )

        experiment_id = lane['experiment_id']

        metrics_dir = get_dev_metrics_dir(experiment_id, lane_id)
        snapshots_file = metrics_dir / "devtrain_snapshots.json"

        if not await async_exists(snapshots_file):
            return RepresentationEvolution(epochs=[])

        snapshots_data = await async_read_json(snapshots_file)

        snapshots_base_dir = get_dev_snapshots_base_dir(experiment_id, lane_id)

        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            compute_evolution_metrics,
            snapshots_data,
            snapshots_base_dir
        )

        return RepresentationEvolution(
            epochs=metrics['epochs'],
            class_separation_over_time=metrics['class_separation'] if any(v is not None for v in metrics['class_separation']) else None,
            best_probe_accuracy_over_time=metrics['probe_accuracy'] if any(v is not None for v in metrics['probe_accuracy']) else None,
            avg_within_class_variance_over_time=metrics['within_class_variance'] if any(v is not None for v in metrics['within_class_variance']) else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting representation evolution: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting evolution data: {str(e)}"
        )


@router.post("/generate-visualizations/{stove_id}")
async def trigger_visualization_generation(stove_id: str, background_tasks: BackgroundTasks):
    """Generate visualizations for all methods and milestones."""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        if stove['stove_type'].upper() != 'DEVTRAIN':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {stove_id} is not a DEVTRAIN job"
            )

        lane_id = stove['lane_id']
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

        dataset_id = experiment['dataset_id']
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )

        dev_snapshots_base_dir = get_dev_snapshots_base_dir(experiment_id, lane_id)
        milestone_dirs = sorted([d for d in dev_snapshots_base_dir.glob("milestone_*") if d.is_dir()],
                               key=lambda x: int(x.name.split("_")[1]))

        if not milestone_dirs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No snapshots found. Training may not have started or completed."
            )

        if stove_id in active_visualization_tasks:
            logger.info(f"Cancelling {len(active_visualization_tasks[stove_id])} existing visualization tasks for job {stove_id}")
            for task in active_visualization_tasks[stove_id]:
                if not task.done():
                    task.cancel()
            active_visualization_tasks[stove_id] = []

        methods = ['deep_dream', 'gradient_saliency', 'feature_maps', 'projection']
        tasks = []
        for method in methods:
            task = asyncio.create_task(
                generate_visualizations_background(
                    stove_id=stove_id,
                    method=method,
                    milestone_dirs=milestone_dirs,
                    dataset=dataset,
                    experiment=experiment
                )
            )
            tasks.append(task)

        active_visualization_tasks[stove_id] = tasks

        logger.info(f"Triggered visualization generation for job {stove_id}, methods: {methods}")

        return {
            "success": True,
            "message": f"Started generating visualizations for {len(methods)} methods across {len(milestone_dirs)} snapshots",
            "methods": methods,
            "num_snapshots": len(milestone_dirs)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering visualization generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger visualization generation: {str(e)}"
        )


@router.post("/reset/{stove_id}")
async def reset_devtrain(stove_id: str):
    """Reset DEVTRAIN: delete training metrics and snapshots, keep the NETINIT configuration."""
    try:
        import shutil

        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        if stove['stove_type'].upper() != 'DEVTRAIN':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {stove_id} is not a DEVTRAIN job"
            )

        lane_id = stove['lane_id']
        lane = await LaneDB.get_by_id(lane_id)
        if not lane:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lane {lane_id} not found"
            )

        experiment_id = lane['experiment_id']
        metrics_dir = get_dev_metrics_dir(experiment_id, lane_id)
        cancellation_file = metrics_dir / "devtrain_cancel.flag"

        await StoveDB.update_status(stove_id, status="ready", progress=0.0)
        await StoveDB.update_configuration(stove_id, {})

        # A cancellation flag stops training gracefully
        logger.info(f"Creating cancellation flag for job {stove_id}")
        cancellation_file.touch()

        dev_snapshots_base_dir = get_dev_snapshots_base_dir(experiment_id, lane_id)

        # Deleting storage makes a running training fail on its next write
        if dev_snapshots_base_dir.exists():
            import shutil
            logger.info(f"Deleting dev_snapshots directory for job {stove_id}")
            shutil.rmtree(dev_snapshots_base_dir)
            dev_snapshots_base_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            # Recreate the cancellation flag, which the rmtree removed
            cancellation_file.touch()
            logger.info(f"Deleted and recreated dev_snapshots for job {stove_id}")

        # snapshot_analysis depends on the DEVTRAIN snapshots
        from ..storage import get_snapshot_analysis_dir
        snapshot_analysis_dir = get_snapshot_analysis_dir(experiment_id, lane_id)
        if snapshot_analysis_dir.exists():
            shutil.rmtree(snapshot_analysis_dir)
            logger.info(f"Deleted snapshot_analysis directory for job {stove_id}")

        if stove_id in active_training_tasks:
            task = active_training_tasks[stove_id]
            if not task.done():
                logger.info(f"Cancelling active training task for job {stove_id}")
                task.cancel()
                try:
                    # Brief wait; training also fails on its next write
                    await asyncio.wait_for(task, timeout=5.0)
                    logger.info(f"Training task stopped gracefully for job {stove_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"Training task did not stop within 5s, it will fail when trying to write files")
                except asyncio.CancelledError:
                    logger.info(f"Training task cancelled for job {stove_id}")
                except Exception as e:
                    logger.warning(f"Error while cancelling training task (expected if files deleted): {e}")

        if stove_id in active_visualization_tasks:
            tasks = active_visualization_tasks[stove_id]
            active_tasks = [t for t in tasks if not t.done()]
            if active_tasks:
                logger.info(f"Cancelling {len(active_tasks)} active visualization tasks for job {stove_id}")
                for task in active_tasks:
                    task.cancel()
                try:
                    await asyncio.wait_for(asyncio.gather(*active_tasks, return_exceptions=True), timeout=2.0)
                    logger.info(f"Visualization tasks cancelled for job {stove_id}")
                except asyncio.TimeoutError:
                    logger.warning(f"Some visualization tasks did not stop within 2s")
                except Exception as e:
                    logger.warning(f"Error while cancelling visualization tasks: {e}")
            active_visualization_tasks[stove_id] = []

        if cancellation_file.exists():
            cancellation_file.unlink()
            logger.info(f"Removed cancellation flag for job {stove_id}")

        logger.info(f"Successfully reset DEVTRAIN job {stove_id}")

        return {
            "success": True,
            "message": "DEVTRAIN data has been reset. You can now start training again.",
            "stove_id": stove_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting DEVTRAIN: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset DEVTRAIN: {str(e)}"
        )


@router.get("/visualizations/{stove_id}/{method}")
async def get_visualizations(
    stove_id: str,
    method: Literal["deep_dream", "gradient_saliency", "feature_maps", "projection"],
    background_tasks: BackgroundTasks
) -> VisualizationsResponse:
    """Visualizations of all snapshots for one method, with per-milestone completion status."""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        if stove['stove_type'].upper() != 'DEVTRAIN':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job {stove_id} is not a DEVTRAIN job"
            )

        lane_id = stove['lane_id']
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

        dataset_id = experiment['dataset_id']
        dataset = get_dataset_by_id(dataset_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {dataset_id} not found"
            )

        def find_class(nodes, target_id):
            for node in nodes:
                if node.id == target_id:
                    return node
                if node.children:
                    result = find_class(node.children, target_id)
                    if result:
                        return result
            return None

        classes_list = []
        for class_id in experiment['selected_classes']:
            class_node = find_class(dataset.hierarchy, class_id)
            if class_node:
                classes_list.append({'name': class_node.name, 'id': class_id})

        dev_snapshots_base_dir = get_dev_snapshots_base_dir(experiment_id, lane_id)
        metrics_dir = get_dev_metrics_dir(experiment_id, lane_id)
        snapshots_data = []

        snapshots_file = metrics_dir / "devtrain_snapshots.json"
        if not await async_exists(snapshots_file):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No snapshots found. Training may not have started."
            )

        all_snapshot_metadata = await async_read_json(snapshots_file)

        snapshot_metadata_map = {s['milestone_index']: s for s in all_snapshot_metadata}

        all_milestone_paths = await async_glob(dev_snapshots_base_dir, "milestone_*")
        milestone_dirs = sorted(
            [d for d in all_milestone_paths if await async_is_dir(d)],
            key=lambda x: int(x.name.split("_")[1])
        )

        total_generated = 0
        total_snapshots = len(milestone_dirs)

        for milestone_dir in milestone_dirs:
            milestone_index = int(milestone_dir.name.split("_")[1])

            snapshot_metadata = snapshot_metadata_map.get(milestone_index)
            if not snapshot_metadata:
                logger.warning(f"Metadata not found for milestone {milestone_index}")
                continue

            epoch = snapshot_metadata.get('epoch', 0)

            vis_dir = milestone_dir / "visualizations" / method

            vis_exists = await async_exists(vis_dir)
            png_files = await async_glob(vis_dir, "*.png") if vis_exists else []

            if vis_exists and png_files:
                status_val = "complete"
                visualizations = []

                for class_idx, class_info in enumerate(classes_list):
                    class_name = class_info['name']

                    if method == "projection":
                        # Projection: one image for all classes
                        img_path = vis_dir / "projection.png"
                        if await async_exists(img_path):
                            image_url = f"/api/devtrain/image/{stove_id}/{milestone_index}/{method}/projection.png"
                            visualizations.append(VisualizationItem(
                                class_idx=class_idx,
                                class_name=class_name,
                                image_url=image_url
                            ))
                        break
                    else:
                        img_path = vis_dir / f"class_{class_idx}.png"
                        if await async_exists(img_path):
                            image_url = f"/api/devtrain/image/{stove_id}/{milestone_index}/{method}/class_{class_idx}.png"
                            visualizations.append(VisualizationItem(
                                class_idx=class_idx,
                                class_name=class_name,
                                image_url=image_url
                            ))

                total_generated += 1
            else:
                status_val = "pending"
                visualizations = []
            
            snapshots_data.append(SnapshotVisualizations(
                milestone_index=milestone_index,
                epoch=epoch,
                visualizations=visualizations,
                status=status_val
            ))

        generation_progress = total_generated / total_snapshots if total_snapshots > 0 else 1.0

        # Generation is started only through the /generate-visualizations endpoint

        return VisualizationsResponse(
            method=method,
            snapshots=snapshots_data,
            generation_progress=generation_progress
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting visualizations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting visualizations: {str(e)}"
        )


@router.get("/image/{stove_id}/{milestone_index}/{method}/{filename}")
async def serve_visualization_image(
    stove_id: str,
    milestone_index: int,
    method: str,
    filename: str
):
    """Serve a visualization image."""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {stove_id} not found"
            )

        lane_id = stove['lane_id']
        lane = await LaneDB.get_by_id(lane_id)
        if not lane:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Lane {lane_id} not found"
            )

        experiment_id = lane['experiment_id']
        
        milestone_dir = get_dev_snapshot_dir(experiment_id, lane_id, milestone_index)
        image_path = milestone_dir / "visualizations" / method / filename

        if not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Visualization image not found: {filename}"
            )

        return FileResponse(
            path=str(image_path),
            media_type="image/png",
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving visualization image: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error serving image: {str(e)}"
        )


async def generate_visualizations_background(
    stove_id: str,
    method: str,
    milestone_dirs: list,
    dataset,
    experiment: dict
):
    """Background task: generate visualizations for snapshots that lack them."""
    try:
        import sys
        from pathlib import Path
        ml_package_path = Path(__file__).parent.parent.parent.parent / "ml" / "src"
        if str(ml_package_path) not in sys.path:
            sys.path.insert(0, str(ml_package_path))

        from representation_visualizations import (
            generate_deep_dream,
            generate_gradient_saliency,
            generate_feature_maps,
            generate_projection_evolution,
            load_representative_samples,
            save_visualization
        )

        logger.info(f"Starting background generation of {method} visualizations for job {stove_id}")

        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            logger.error(f"Job {stove_id} not found")
            return

        lane = await LaneDB.get_by_id(stove['lane_id'])
        if not lane:
            logger.error(f"Lane not found for job {stove_id}")
            return

        experiment_id = lane['experiment_id']
        lane_id = lane['id']
        metrics_dir = get_dev_metrics_dir(experiment_id, lane_id)

        snapshots_file = metrics_dir / "devtrain_snapshots.json"
        if not snapshots_file.exists():
            logger.error(f"Snapshots metadata file not found: {snapshots_file}")
            return

        with open(snapshots_file, 'r') as f:
            all_snapshot_metadata = json.load(f)

        snapshot_metadata_map = {s['milestone_index']: s for s in all_snapshot_metadata}

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        selected_classes = experiment['selected_classes']
        num_classes = len(selected_classes)

        def find_class(nodes, target_id):
            for node in nodes:
                if node.id == target_id:
                    return node
                if node.children:
                    result = find_class(node.children, target_id)
                    if result:
                        return result
            return None

        # NETINIT configuration for model reconstruction
        stoves_in_lane = await StoveDB.get_all(lane_id)
        netinit_stove = next((s for s in stoves_in_lane if s['stove_type'].upper() == 'NETINIT'), None)
        if not netinit_stove or not netinit_stove['configuration']:
            logger.error(f"NETINIT configuration not found for lane {lane_id}")
            return

        from ..netinit_models import NetInitConfiguration as NetInitConfigPydantic
        netinit_config_pydantic = NetInitConfigPydantic.model_validate(netinit_stove['configuration'])

        # Projection: collect the mean activations of all snapshots first
        if method == "projection":
            snapshots_activations = []
            class_names = []
            for class_id in selected_classes:
                class_node = find_class(dataset.hierarchy, class_id)
                if class_node:
                    class_names.append(class_node.name)

            for milestone_dir in milestone_dirs:
                milestone_index = int(milestone_dir.name.split("_")[1])

                snapshot_metadata = snapshot_metadata_map.get(milestone_index)
                if not snapshot_metadata:
                    logger.warning(f"Metadata not found for milestone {milestone_index}")
                    continue

                epoch = snapshot_metadata.get('epoch', 0)

                activations_path = milestone_dir / "mean_activations.pt"
                if activations_path.exists():
                    activations = torch.load(activations_path, map_location='cpu')
                    snapshots_activations.append({
                        'milestone_index': milestone_index,
                        'epoch': epoch,
                        'activations': activations
                    })

            # t-SNE is CPU-intensive
            if snapshots_activations:
                projection_img = await asyncio.to_thread(
                    generate_projection_evolution,
                    snapshots_data=snapshots_activations,
                    method='tsne',
                    class_names=class_names
                )

                # The same image is saved for each milestone
                for milestone_dir in milestone_dirs:
                    vis_dir = milestone_dir / "visualizations" / "projection"
                    await asyncio.to_thread(
                        save_visualization,
                        img_array=projection_img,
                        save_path=vis_dir / "projection.png"
                    )

        else:
            # Per-class methods (deep_dream, gradient_saliency, feature_maps)
            total_work = len(milestone_dirs) * num_classes
            completed_work = 0

            for idx, milestone_dir in enumerate(milestone_dirs):
                if asyncio.current_task().cancelled():
                    logger.info(f"Visualization task cancelled at milestone {idx}")
                    raise asyncio.CancelledError()

                milestone_index = int(milestone_dir.name.split("_")[1])

                vis_dir = milestone_dir / "visualizations" / method
                if vis_dir.exists() and any(vis_dir.glob("*.png")):
                    completed_work += num_classes
                    continue

                checkpoint_path = milestone_dir / "checkpoint.pt"
                if not checkpoint_path.exists():
                    logger.warning(f"Checkpoint not found for milestone {milestone_index}")
                    completed_work += num_classes
                    continue

                # The checkpoint holds the state_dict only
                state_dict = torch.load(checkpoint_path, map_location=device)

                try:
                    from models import create_network

                    model = create_network(
                        network_type=netinit_config_pydantic.network_type,
                        num_classes=num_classes,
                        pretrained=False
                    )

                    model.load_state_dict(state_dict)
                    model = model.to(device)
                    model.eval()

                except Exception as e:
                    logger.error(f"Error loading model for milestone {milestone_index}: {e}", exc_info=True)
                    completed_work += num_classes
                    continue

                for class_idx in range(num_classes):
                    class_id = selected_classes[class_idx]
                    class_node = find_class(dataset.hierarchy, class_id)
                    class_name = class_node.name if class_node else f"Class {class_idx}"

                    try:
                        # Generation runs in a thread to keep the event loop free
                        if method == "deep_dream":
                            img = await asyncio.to_thread(
                                generate_deep_dream,
                                model=model,
                                class_idx=class_idx,
                                device=device,
                                num_iterations=50,  # kept low to limit memory use
                                learning_rate=0.05
                            )
                            torch.cuda.empty_cache()
                        elif method == "gradient_saliency":
                            # Random input as a stand-in sample
                            representative_sample = torch.randn(3, 224, 224)
                            img = await asyncio.to_thread(
                                generate_gradient_saliency,
                                model=model,
                                class_idx=class_idx,
                                representative_sample=representative_sample,
                                device=device
                            )
                        elif method == "feature_maps":
                            representative_sample = torch.randn(3, 224, 224)
                            img = await asyncio.to_thread(
                                generate_feature_maps,
                                model=model,
                                class_idx=class_idx,
                                representative_sample=representative_sample,
                                device=device
                            )

                        save_path = vis_dir / f"class_{class_idx}.png"
                        await asyncio.to_thread(
                            save_visualization,
                            img_array=img,
                            save_path=save_path,
                            metadata={
                                'class_idx': class_idx,
                                'class_name': class_name,
                                'method': method,
                                'milestone_index': milestone_index
                            }
                        )

                    except Exception as e:
                        logger.error(f"Error generating {method} for class {class_idx} at milestone {milestone_index}: {e}")

                    completed_work += 1
                    progress = completed_work / total_work
                    await ConnectionManager.broadcast(
                        stove_id,
                        DevTrainWebSocketMessage(
                            message_type="visualization_progress",
                            stove_id=stove_id,
                            progress=progress
                        )
                    )
        
        logger.info(f"Completed background generation of {method} visualizations for job {stove_id}")

        await ConnectionManager.broadcast(
            stove_id,
            DevTrainWebSocketMessage(
                message_type="visualization_complete",
                stove_id=stove_id
            )
        )

    except asyncio.CancelledError:
        logger.info(f"Visualization generation ({method}) cancelled for job {stove_id}")
        # No error message for an intentional cancellation
        raise

    except Exception as e:
        logger.error(f"Error in background visualization generation: {e}", exc_info=True)
        await ConnectionManager.broadcast(
            stove_id,
            DevTrainWebSocketMessage(
                message_type="error",
                stove_id=stove_id,
                error_message=f"Visualization generation failed: {str(e)}"
            )
        )
