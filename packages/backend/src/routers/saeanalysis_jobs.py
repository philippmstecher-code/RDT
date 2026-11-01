from __future__ import annotations

"""
SAE Analysis (SAEANALYSIS) router for the per-checkpoint SAE approach.

Serves the 5-tab view hierarchy:
  Process View → Superclass View → Class View → Sample View → Summary
"""
import logging
import asyncio
import json
import shutil
import base64
import io
from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, status, BackgroundTasks
from fastapi.responses import JSONResponse
import torch
import numpy as np
import torchvision.transforms as transforms

from ..saeanalysis_stove_models import (
    SAEAnalysisDashboard,
    StartSAEAnalysisRequest,
    StartSAEAnalysisResponse,
    ResetSAEAnalysisRequest,
    ResetSAEAnalysisResponse,
    ReclassifySAEAnalysisRequest,
    ReclassifySAEAnalysisResponse,
    SAEAnalysisWebSocketMessage,
    ProcessViewData,
    FeatureLandscapePoint,
    ProcessIntensityPoint,
    FeatureMatchingPoint,
    AllSuperclassesViewData,
    SuperclassViewData,
    SSIEvolutionPoint,
    CSIEvolutionPoint,
    ProcessCounts,
    ClassViewData,
    ClassProcessSummary,
    SampleConsistencyData,
    KeyFeatureInfo,
    SampleEventStats,
    SuperclassParticipation,
    FeatureDirectoryEntry,
    SampleCheckpointData,
    SampleTransitionData,
    SampleDetailResponse,
    SampleLayerActivity,
    FeatureFlowData,
    SummaryViewData,
    HypothesisVerdict,
    NullBaselinePoint,
    SelectivityEvolutionPoint,
    WithinCheckpointControl,
    ControlCheckpointResult,
    ControlLayerResult,
    SampleImageResponse,
    TerminalSimilarityPoint,
    SampleTerminalSimilarityResponse,
    CatalogFeatureEntry,
    FeatureCatalogResponse,
    ProcessCatalogEntry,
    ProcessCatalogResponse,
    InsightsResponse,
    InsightsDatasetOverview,
    BoundaryCrystallizationData,
    BoundaryCrystallizationPoint,
    TemporalOrderingData,
    TemporalOrderingEntry,
    FeatureGenealogyData,
    FeatureLineage,
    LineageNode,
    CrossLayerCompositionData,
    CrossLayerSnapshot,
    CrossLayerPair,
    InformationBottleneckData,
    InformationBottleneckPoint,
    SuperpositionEvolutionData,
    SuperpositionEvolutionPoint,
    ForgettableExamplesData,
    ForgettableExampleEntry,
    AcquiredEquivalenceData,
    AcquiredEquivalencePoint,
    LayerAbstractionProgressionData,
    LayerAbstractionPoint,
    LayerAbstractionGradient,
    TaskGeneralAbstractionData,
    TaskGeneralSaiPoint,
    TaskGeneralLayerProfile,
    TaskGeneralOnsetComparison,
    TaskGeneralPersistence,
    GranularProcessEntry,
    GranularProcessCatalogResponse,
    CriticalPeriodData,
    CriticalPeriodPoint,
    GeometryConvergenceData,
    RDMPoint,
    ProcessInterferenceData,
    ProcessInterferencePoint,
    CatastrophicForgettingData,
    ForgettingTransitionPoint,
    PolysemantricityDynamicsData,
    PolysemanticityCkptPoint,
    SharpeningTransitionPoint,
    BornPIProfile,
    LayerDepthPIPoint,
    RepresentationalDriftData,
    DriftTransitionPoint,
    LayerDriftPoint,
    DatasetOverviewResponse,
    ProcessClassificationBreakdown,
    FeatureSurvivalData,
    SurvivalCurvePoint,
    SurvivalCohort,
    LayerSurvivalSummary,
)
from ..storage import get_saeanalysis_dir, get_lane_dir, get_dev_snapshots_base_dir, get_terminal_analysis_dir
from ..database import StoveDB, LaneDB, ExperimentDB
from ..datasets import get_class_thumbnail_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/saeanalysis-stove", tags=["saeanalysis_jobs"])


def _safe_layer_filename(layer_name: str) -> str:
    """Replace dots with underscores for filesystem-safe layer names (mirrors ML code)."""
    return layer_name.replace(".", "_")

active_connections: dict[str, list[WebSocket]] = {}


class ConnectionManager:
    """Manage WebSocket connections for live SAEANALYSIS updates"""

    @staticmethod
    async def connect(stove_id: str, websocket: WebSocket):
        await websocket.accept()
        if stove_id not in active_connections:
            active_connections[stove_id] = []
        active_connections[stove_id].append(websocket)
        logger.info(f"SAEANALYSIS WebSocket connected for job {stove_id}")

    @staticmethod
    async def disconnect(stove_id: str, websocket: WebSocket):
        if stove_id in active_connections:
            if websocket in active_connections[stove_id]:
                active_connections[stove_id].remove(websocket)
            if not active_connections[stove_id]:
                del active_connections[stove_id]
        logger.info(f"SAEANALYSIS WebSocket disconnected for job {stove_id}")

    @staticmethod
    async def broadcast(stove_id: str, message: SAEAnalysisWebSocketMessage):
        if stove_id not in active_connections:
            return

        message_json = message.model_dump_json()
        disconnected = []

        for websocket in active_connections[stove_id]:
            try:
                await websocket.send_text(message_json)
            except Exception as e:
                logger.error(f"Error broadcasting to SAEANALYSIS WebSocket: {e}")
                disconnected.append(websocket)

        for websocket in disconnected:
            await ConnectionManager.disconnect(stove_id, websocket)


@router.websocket("/ws/{stove_id}")
async def websocket_endpoint(websocket: WebSocket, stove_id: str):
    """WebSocket endpoint for live SAEANALYSIS progress updates"""
    await ConnectionManager.connect(stove_id, websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ConnectionManager.disconnect(stove_id, websocket)
    except Exception:
        await ConnectionManager.disconnect(stove_id, websocket)



async def _load_sae_context(stove_id: str):
    """Helper to load job, lane, experiment, and sae_results.json for a completed SAEANALYSIS job."""
    stove = await StoveDB.get_by_id(stove_id)
    if not stove:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {stove_id} not found"
        )

    if stove['stove_type'].upper() != 'SAEANALYSIS':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job {stove_id} is not a SAEANALYSIS job"
        )

    if stove['status'] != 'completed':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SAEANALYSIS must be completed before accessing detail data"
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

    sae_dir = get_saeanalysis_dir(experiment_id, lane_id)
    results_file = sae_dir / "sae_results.json"

    sae_data = None
    if results_file.exists():
        with open(results_file, 'r') as f:
            sae_data = json.load(f)

    return stove, lane, experiment, sae_dir, sae_data


def _resolve_method_data(sae_data: dict, matching_method: str) -> dict:
    """Return the sae_data dict with matching-method-specific fields resolved.

    For 'activation' (default), top-level keys are used directly.
    For 'weight', fields are overlaid from sae_data['weight_matching'].
    """
    if not sae_data:
        return sae_data
    if matching_method == "weight" and "weight_matching" in sae_data:
        wm = sae_data["weight_matching"]
        resolved = dict(sae_data)
        for key in ("feature_matching", "feature_landscape",
                     "class_process_summary", "superclass_summary",
                     "sample_consistency", "process_intensity",
                     "hypotheses", "null_baseline",
                     "selectivity_evolution", "discrimination_gradients"):
            if key in wm:
                resolved[key] = wm[key]
        return resolved
    return sae_data


def _transitions_dir(sae_dir: Path, matching_method: str) -> Path:
    """Return the transitions directory for the given matching method."""
    if matching_method == "weight":
        return sae_dir / "transitions_weight"
    return sae_dir / "transitions"


def _enrich_pi_with_training_metadata(
    pi_list: list,
    experiment_id: str,
    lane_id: str,
) -> None:
    """Enrich ProcessIntensityPoint list with val_accuracy and weight_updates_delta from devtrain_snapshots.json."""
    snapshots_file = get_dev_snapshots_base_dir(experiment_id, lane_id) / "metrics" / "devtrain_snapshots.json"
    if not snapshots_file.exists():
        return
    try:
        with open(snapshots_file, 'r') as f:
            snapshots = json.load(f)
        if not snapshots or not isinstance(snapshots, list):
            return
        snap_by_idx = {s['milestone_index']: s for s in snapshots}
        # Transition labels are "0->1", "1->2", ... (the arrow may also be the unicode character)
        for pi_point in pi_list:
            t = pi_point.transition
            parts = t.replace('→', '->').split('->')
            if len(parts) != 2:
                continue
            try:
                from_idx = int(parts[0].strip())
                to_part = parts[1].strip()
                if to_part == 'terminal':
                    to_idx = max(snap_by_idx.keys()) if snap_by_idx else None
                else:
                    to_idx = int(to_part)
            except (ValueError, TypeError):
                continue
            if to_idx is not None and to_idx in snap_by_idx:
                to_snap = snap_by_idx[to_idx]
                pi_point.end_val_accuracy = to_snap.get('val_accuracy')
                from_snap = snap_by_idx.get(from_idx)
                if from_snap is not None:
                    wu_to = to_snap.get('weight_updates', 0)
                    wu_from = from_snap.get('weight_updates', 0)
                    pi_point.weight_updates_delta = wu_to - wu_from
    except Exception:
        pass  # The enrichment is optional


def _compute_stable_adjustments(
    sae_dir: Path,
    layers: list[str],
    checkpoint_labels: list[str],
    matching_method: str,
    selectivity: dict,
) -> dict:
    """Count stable features labelled ab_h/tg_h/di_h, to subtract them when only
    events on born and died features are shown.

    Returns {
        'global': {transition: {'ab_h': N, 'tg_h': N, 'di_h': N}},
        'per_class': {class_idx: {transition: {'ab_h': N, 'tg_h': N, 'di_h': N}}},
        'per_class_total': {class_idx: {'ab_h': N, 'tg_h': N, 'di_h': N, 'stable': N}},
    }
    """
    transitions_dir = _transitions_dir(sae_dir, matching_method)
    transitions_list: list[str] = []
    for i in range(len(checkpoint_labels) - 1):
        transitions_list.append(f"{checkpoint_labels[i]}->{checkpoint_labels[i+1]}")

    global_adj: dict[str, dict[str, int]] = {}
    per_class_adj: dict[int, dict[str, dict[str, int]]] = {}
    per_class_total: dict[int, dict[str, int]] = {}

    for pair_key in transitions_list:
        safe_pair = pair_key.replace("->", "_to_")
        global_adj[pair_key] = {'ab_h': 0, 'tg_h': 0, 'di_h': 0, 'stable': 0}

        for layer_name in layers:
            safe_layer = layer_name.replace(".", "_")
            trans_file = transitions_dir / safe_pair / f"{safe_layer}.json"
            if not trans_file.exists():
                continue

            try:
                with open(trans_file) as f:
                    sample_events = json.load(f)
            except Exception:
                continue

            for ev in sample_events:
                ci = ev.get('class_idx', 0)
                if ci not in per_class_adj:
                    per_class_adj[ci] = {}
                    per_class_total[ci] = {'ab_h': 0, 'tg_h': 0, 'di_h': 0, 'stable': 0}
                if pair_key not in per_class_adj[ci]:
                    per_class_adj[ci][pair_key] = {'ab_h': 0, 'tg_h': 0, 'di_h': 0, 'stable': 0}

                for s in ev.get('stable', []):
                    global_adj[pair_key]['stable'] += 1
                    per_class_adj[ci][pair_key]['stable'] += 1
                    per_class_total[ci]['stable'] += 1

                    # Process labels as stored by the ML classification
                    # (already gated on selectivity increase)
                    for proc in s.get('processes', []):
                        if proc in ('ab_h', 'tg_h', 'di_h'):
                            global_adj[pair_key][proc] += 1
                            per_class_adj[ci][pair_key][proc] += 1
                            per_class_total[ci][proc] += 1

    return {
        'global': global_adj,
        'per_class': per_class_adj,
        'per_class_total': per_class_total,
    }


async def _load_sample_image_base64(experiment: dict, class_idx: int, sample_idx: int) -> str:
    """Load a sample image from the validation dataset and return as base64 data URI."""
    import sys
    ml_package_path = Path(__file__).parent.parent.parent.parent / "ml" / "src"
    if str(ml_package_path) not in sys.path:
        sys.path.insert(0, str(ml_package_path))

    from training import get_dataset

    val_dataset, _ = await asyncio.get_event_loop().run_in_executor(
        None,
        get_dataset,
        experiment['dataset_id'],
        experiment['selected_classes'],
        False,  # train=False (validation set)
        experiment['id']
    )

    class_samples = []
    for i, (img, label) in enumerate(val_dataset):
        if label == class_idx:
            class_samples.append((img, i))
        if len(class_samples) > sample_idx:
            break

    if sample_idx >= len(class_samples):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample {sample_idx} not found for class {class_idx}"
        )

    img_tensor, _ = class_samples[sample_idx]

    dataset_id = experiment['dataset_id']
    if dataset_id in ['cifar10', 'cifar100']:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    else:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img_denorm = img_tensor * std + mean
    img_denorm = torch.clamp(img_denorm, 0, 1)

    to_pil = transforms.ToPILImage()
    img_pil = to_pil(img_denorm)

    buffer = io.BytesIO()
    img_pil.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_base64}"


# Dashboard (lightweight status)

@router.get("/dashboard/{stove_id}", response_model=SAEAnalysisDashboard)
async def get_dashboard(stove_id: str):
    """Get lightweight dashboard status for SAEANALYSIS job."""
    try:
        stove = await StoveDB.get_by_id(stove_id)
        if not stove:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {stove_id} not found")

        if stove['stove_type'].upper() != 'SAEANALYSIS':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job {stove_id} is not a SAEANALYSIS job")

        lane_id = stove['lane_id']
        lane = await LaneDB.get_by_id(lane_id)
        if not lane:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Lane {lane_id} not found")

        experiment = await ExperimentDB.get_by_id(lane['experiment_id'])
        if not experiment:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Experiment not found")

        dashboard = SAEAnalysisDashboard(
            status=stove['status'],
            progress=stove['progress']
        )

        sae_dir = get_saeanalysis_dir(lane['experiment_id'], lane_id)
        progress_file = sae_dir / "saeanalysis_progress.json"
        if progress_file.exists():
            try:
                prog_data = json.loads(progress_file.read_text())
                dashboard.current_step = prog_data.get("message", "")
            except Exception:
                pass

        if stove['status'] == 'completed':
            results_file = sae_dir / "sae_results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    data = json.load(f)
                metadata = data.get('metadata', {})
                dashboard.approach = metadata.get('approach', '')
                dashboard.layers_analyzed = metadata.get('layers', [])
                dashboard.checkpoint_labels = metadata.get('checkpoint_labels', [])
                dashboard.n_classes = metadata.get('n_classes', 0)
                dashboard.selected_classes = metadata.get('selected_classes', [])
                dashboard.matching_methods = metadata.get('matching_methods', ['activation'])

        return dashboard

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dashboard: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Process View

@router.get("/process-view/{stove_id}", response_model=ProcessViewData)
async def get_process_view(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                           exclude_stable: bool = Query(False, description="Exclude stable features from process counts")):
    """Get data for the Process View tab: feature landscape + process intensity + feature matching."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)
        sae_data_raw_for_adj = sae_data
        sae_data = _resolve_method_data(sae_data, matching_method)

        if not sae_data:
            return ProcessViewData()

        metadata = sae_data.get('metadata', {})
        layers = metadata.get('layers', [])
        checkpoint_labels = metadata.get('checkpoint_labels', [])

        adj_global: dict[str, dict[str, int]] = {}
        if exclude_stable:
            selectivity = sae_data_raw_for_adj.get('selectivity', sae_data.get('selectivity', {}))
            adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
            adj_global = adjs['global']

        # Feature landscape: checkpoint x layer grid
        feature_landscape: list[FeatureLandscapePoint] = []
        landscape_data = sae_data.get('feature_landscape', {})
        for ckpt_label, layer_dict in landscape_data.items():
            if not isinstance(layer_dict, dict):
                continue
            for layer_name, vals in layer_dict.items():
                if not isinstance(vals, dict):
                    continue
                feature_landscape.append(FeatureLandscapePoint(
                    checkpoint_label=ckpt_label,
                    layer_name=layer_name,
                    n_alive=vals.get('n_alive', 0),
                    mean_ssi=vals.get('mean_ssi', 0.0),
                    mean_csi=vals.get('mean_csi', 0.0),
                    mean_entropy=vals.get('mean_entropy', 0.0),
                    mean_sai=vals.get('mean_sai', 0.0),
                    n_high_ssi=vals.get('n_high_ssi', 0),
                    n_high_sai=vals.get('n_high_sai', 0),
                ))

        # Process intensity timeline
        process_intensity: list[ProcessIntensityPoint] = []
        for pi in sae_data.get('process_intensity', []):
            if not isinstance(pi, dict):
                continue
            t = pi.get('transition', '')
            adj = adj_global.get(t, {}) if exclude_stable else {}
            ab_h = max(0, pi.get('ab_h', 0) - adj.get('ab_h', 0))
            tg_h = max(0, pi.get('tg_h', 0) - adj.get('tg_h', 0))
            di_h = max(0, pi.get('di_h', 0) - adj.get('di_h', 0))
            as_h = pi.get('as_h', 0)
            de_h = pi.get('de_h', 0)
            # tg_h is a subcategory of ab_h and is not added to the total
            total = ab_h + di_h + as_h + de_h + pi.get('unclassified', 0)
            proc_counts = {'ab_h': ab_h, 'di_h': di_h, 'as_h': as_h, 'de_h': de_h}
            dominant = max(proc_counts, key=proc_counts.get) if total > 0 else 'none'  # type: ignore
            process_intensity.append(ProcessIntensityPoint(
                transition=t,
                ab_h=ab_h,
                tg_h=tg_h,
                di_h=di_h,
                as_h=as_h,
                de_h=de_h,
                unclassified=pi.get('unclassified', 0),
                total=total,
                churn=pi.get('churn', 0.0),
                dominant=dominant,
            ))

        # Feature matching
        feature_matching: list[FeatureMatchingPoint] = []
        matching_data = sae_data.get('feature_matching', {})
        for pair_key, layer_dict in matching_data.items():
            if not isinstance(layer_dict, dict):
                continue
            for layer_name, vals in layer_dict.items():
                if not isinstance(vals, dict):
                    continue
                feature_matching.append(FeatureMatchingPoint(
                    transition=pair_key,
                    layer_name=layer_name,
                    n_stable=vals.get('n_stable', 0),
                    n_born=vals.get('n_born', 0),
                    n_died=vals.get('n_died', 0),
                    n_transformed=vals.get('n_transformed', 0),
                ))

        return ProcessViewData(
            checkpoint_labels=checkpoint_labels,
            layers=layers,
            feature_landscape=feature_landscape,
            process_intensity=process_intensity,
            feature_matching=feature_matching,
            reconstruction_quality=sae_data.get('reconstruction_quality', {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting process view: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Superclass View

@router.get("/superclass-view/{stove_id}", response_model=AllSuperclassesViewData)
async def get_superclass_view(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                              exclude_stable: bool = Query(False, description="Exclude stable features from process counts")):
    """Get data for the Superclass View tab."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)
        sae_data_raw_for_adj = sae_data
        sae_data = _resolve_method_data(sae_data, matching_method)

        if not sae_data:
            return AllSuperclassesViewData()

        sc_summary = sae_data.get('superclass_summary', {})
        superclass_names = sorted(sc_summary.keys())
        superclasses: dict[str, SuperclassViewData] = {}

        metadata = sae_data.get('metadata', {})
        selected_classes = metadata.get('selected_classes', experiment.get('selected_classes', []))
        class_name_to_idx = {name: i for i, name in enumerate(selected_classes)}
        class_process_summary = sae_data.get('class_process_summary', {})

        # Stable adjustments
        adj_per_class: dict[int, dict[str, dict[str, int]]] = {}
        adj_per_class_total: dict[int, dict[str, int]] = {}
        if exclude_stable:
            layers = metadata.get('layers', [])
            checkpoint_labels = metadata.get('checkpoint_labels', [])
            selectivity = sae_data_raw_for_adj.get('selectivity', sae_data.get('selectivity', {}))
            adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
            adj_per_class = adjs['per_class']
            adj_per_class_total = adjs['per_class_total']

        n_samples_per_class = 0
        activations_dir = sae_dir / "activations"
        if activations_dir.exists():
            for ckpt_dir in activations_dir.iterdir():
                if ckpt_dir.is_dir():
                    for act_file in ckpt_dir.glob("*.npy"):
                        import numpy as np
                        H = np.load(str(act_file))
                        n_classes = max(len(selected_classes), 1)
                        n_samples_per_class = H.shape[0] // n_classes
                        break
                    break

        for sc_name, sc_data in sc_summary.items():
            if not isinstance(sc_data, dict):
                continue

            pb = sc_data.get('process_breakdown', {})
            ssi_evo = [
                SSIEvolutionPoint(checkpoint=pt.get('checkpoint', ''), mean_ssi=pt.get('mean_ssi', 0.0))
                for pt in sc_data.get('ssi_evolution', [])
                if isinstance(pt, dict)
            ]

            # Per-superclass process intensity from the class per_transition data
            per_class_process = sc_data.get('per_class_process', {})
            sc_class_indices = [class_name_to_idx[cn] for cn in per_class_process if cn in class_name_to_idx]
            sc_n_samples = n_samples_per_class * len(sc_class_indices)

            transition_totals: dict[str, dict[str, int]] = {}  # transition -> {ab_h, di_h, ...}
            for ci in sc_class_indices:
                cps = class_process_summary.get(str(ci), {})
                for tr in cps.get('per_transition', []):
                    if not isinstance(tr, dict):
                        continue
                    t_key = tr.get('transition', '')
                    if t_key not in transition_totals:
                        transition_totals[t_key] = {'ab_h': 0, 'tg_h': 0, 'di_h': 0, 'as_h': 0, 'de_h': 0, 'unclassified': 0}
                    for proc in ['ab_h', 'tg_h', 'di_h', 'as_h', 'de_h', 'unclassified']:
                        transition_totals[t_key][proc] += tr.get(proc, 0)

            # Subtract stable-feature contributions from ab_h/di_h
            if exclude_stable:
                for ci in sc_class_indices:
                    ci_adj = adj_per_class.get(ci, {})
                    for t_key, adj_vals in ci_adj.items():
                        if t_key in transition_totals:
                            transition_totals[t_key]['ab_h'] = max(0, transition_totals[t_key]['ab_h'] - adj_vals.get('ab_h', 0))
                            transition_totals[t_key]['tg_h'] = max(0, transition_totals[t_key]['tg_h'] - adj_vals.get('tg_h', 0))
                            transition_totals[t_key]['di_h'] = max(0, transition_totals[t_key]['di_h'] - adj_vals.get('di_h', 0))

            sc_process_intensity: list[ProcessIntensityPoint] = []
            for t_key in sorted(transition_totals.keys()):
                counts = transition_totals[t_key]
                # tg_h is a subcategory of ab_h and is not added to the total
                total = counts['ab_h'] + counts['di_h'] + counts['as_h'] + counts['de_h'] + counts['unclassified']
                dominant = 'none'
                if total > 0:
                    proc_counts = {k: counts[k] for k in ['ab_h', 'di_h', 'as_h', 'de_h']}
                    dominant = max(proc_counts, key=proc_counts.get)  # type: ignore
                sc_process_intensity.append(ProcessIntensityPoint(
                    transition=t_key,
                    ab_h=counts['ab_h'],
                    tg_h=counts['tg_h'],
                    di_h=counts['di_h'],
                    as_h=counts['as_h'],
                    de_h=counts['de_h'],
                    unclassified=counts['unclassified'],
                    total=total,
                    churn=0.0,
                    dominant=dominant,
                ))

            sc_adj_ab_h = sum(adj_per_class_total.get(ci, {}).get('ab_h', 0) for ci in sc_class_indices) if exclude_stable else 0
            sc_adj_tg_h = sum(adj_per_class_total.get(ci, {}).get('tg_h', 0) for ci in sc_class_indices) if exclude_stable else 0
            sc_adj_di_h = sum(adj_per_class_total.get(ci, {}).get('di_h', 0) for ci in sc_class_indices) if exclude_stable else 0
            sc_adj_stable = sum(adj_per_class_total.get(ci, {}).get('stable', 0) for ci in sc_class_indices) if exclude_stable else 0
            pb_ab_h = max(0, pb.get('ab_h', 0) - sc_adj_ab_h)
            pb_tg_h = max(0, pb.get('tg_h', 0) - sc_adj_tg_h)
            pb_di_h = max(0, pb.get('di_h', 0) - sc_adj_di_h)
            # tg_h is a subcategory of ab_h and is not added to the total
            pb_total = pb_ab_h + pb_di_h + pb.get('as_h', 0) + pb.get('de_h', 0) + pb.get('unclassified', 0)

            superclasses[sc_name] = SuperclassViewData(
                superclass_name=sc_name,
                process_breakdown=ProcessCounts(
                    ab_h=pb_ab_h,
                    tg_h=pb_tg_h,
                    di_h=pb_di_h,
                    as_h=pb.get('as_h', 0),
                    de_h=pb.get('de_h', 0),
                    unclassified=pb.get('unclassified', 0),
                    total=pb_total if exclude_stable else pb.get('total', 0),
                ),
                ssi_evolution=ssi_evo,
                per_class_csi_evolution={
                    cn: [CSIEvolutionPoint(checkpoint=pt.get('checkpoint', ''), mean_csi=pt.get('mean_csi', 0.0))
                         for pt in pts if isinstance(pt, dict)]
                    for cn, pts in sc_data.get('per_class_csi_evolution', {}).items()
                },
                per_class_process=per_class_process,
                pairwise_overlap=sc_data.get('pairwise_overlap', {}),
                feature_sharing_breadth=sc_data.get('feature_sharing_breadth', {}),
                shared_feature_growth=sc_data.get('shared_feature_growth', []),
                process_intensity=sc_process_intensity,
                n_samples=sc_n_samples,
            )

        return AllSuperclassesViewData(
            superclass_names=superclass_names,
            superclasses=superclasses,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting superclass view: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Class View

@router.get("/class-view/{stove_id}/{class_idx}", response_model=ClassViewData)
async def get_class_view(stove_id: str, class_idx: int, matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                         exclude_stable: bool = Query(False, description="Exclude stable features from process counts")):
    """Get data for a single class in the Class View tab."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)
        sae_data_raw_for_adj = sae_data
        sae_data = _resolve_method_data(sae_data, matching_method)

        class_name = ""
        if class_idx < len(experiment['selected_classes']):
            class_name = experiment['selected_classes'][class_idx]

        if not sae_data:
            return ClassViewData(class_idx=class_idx, class_name=class_name)

        metadata = sae_data.get('metadata', {})
        layers = metadata.get('layers', [])
        checkpoint_labels = metadata.get('checkpoint_labels', [])

        ci_adj_total: dict[str, int] = {}
        ci_adj_per_trans: dict[str, dict[str, int]] = {}
        if exclude_stable:
            selectivity = sae_data_raw_for_adj.get('selectivity', sae_data.get('selectivity', {}))
            adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
            ci_adj_total = adjs['per_class_total'].get(class_idx, {})
            ci_adj_per_trans = adjs['per_class'].get(class_idx, {})

        # Class process summary
        cps_raw = sae_data.get('class_process_summary', {}).get(str(class_idx), {})
        raw_ab_h = max(0, cps_raw.get('ab_h', 0) - ci_adj_total.get('ab_h', 0)) if exclude_stable else cps_raw.get('ab_h', 0)
        raw_tg_h = max(0, cps_raw.get('tg_h', 0) - ci_adj_total.get('tg_h', 0)) if exclude_stable else cps_raw.get('tg_h', 0)
        raw_di_h = max(0, cps_raw.get('di_h', 0) - ci_adj_total.get('di_h', 0)) if exclude_stable else cps_raw.get('di_h', 0)
        # tg_h is a subcategory of ab_h and is not added to the total
        raw_total = raw_ab_h + raw_di_h + cps_raw.get('as_h', 0) + cps_raw.get('de_h', 0) + cps_raw.get('unclassified', 0)
        proc_map = {'ab_h': raw_ab_h, 'di_h': raw_di_h, 'as_h': cps_raw.get('as_h', 0), 'de_h': cps_raw.get('de_h', 0)}
        raw_dominant = max(proc_map, key=proc_map.get) if raw_total > 0 else 'none'  # type: ignore

        # Adjust per_transition data
        per_transition_adj = cps_raw.get('per_transition', [])
        if exclude_stable and per_transition_adj:
            adjusted_pt = []
            for tr in per_transition_adj:
                if not isinstance(tr, dict):
                    adjusted_pt.append(tr)
                    continue
                t_key = tr.get('transition', '')
                t_adj = ci_adj_per_trans.get(t_key, {})
                adjusted_pt.append({
                    **tr,
                    'ab_h': max(0, tr.get('ab_h', 0) - t_adj.get('ab_h', 0)),
                    'tg_h': max(0, tr.get('tg_h', 0) - t_adj.get('tg_h', 0)),
                    'di_h': max(0, tr.get('di_h', 0) - t_adj.get('di_h', 0)),
                })
            per_transition_adj = adjusted_pt

        process_summary = ClassProcessSummary(
            class_idx=class_idx,
            class_name=cps_raw.get('class_name', class_name),
            ab_h=raw_ab_h,
            tg_h=raw_tg_h,
            di_h=raw_di_h,
            as_h=cps_raw.get('as_h', 0),
            de_h=cps_raw.get('de_h', 0),
            unclassified=cps_raw.get('unclassified', 0),
            stable=cps_raw.get('stable', 0),
            total=raw_total if exclude_stable else cps_raw.get('total', 0),
            dominant=raw_dominant if exclude_stable else cps_raw.get('dominant', 'none'),
            per_transition=per_transition_adj,
        )

        # Sample consistency
        sc_raw = sae_data.get('sample_consistency', {}).get(str(class_idx), {})
        sample_consistency = SampleConsistencyData(
            class_idx=class_idx,
            ab_h_dominant=sc_raw.get('ab_h_dominant', 0),
            tg_h_dominant=sc_raw.get('tg_h_dominant', 0),
            di_h_dominant=sc_raw.get('di_h_dominant', 0),
            as_h_dominant=sc_raw.get('as_h_dominant', 0),
            de_h_dominant=sc_raw.get('de_h_dominant', 0),
            mixed=sc_raw.get('mixed', 0),
        )

        # Mean SSI/CSI per checkpoint
        selectivity = sae_data.get('selectivity', {})
        mean_ssi_per_ckpt: dict[str, float] = {}
        mean_csi_per_ckpt: dict[str, float] = {}
        for ckpt_label in checkpoint_labels:
            ckpt_sel = selectivity.get(ckpt_label, {})
            ssi_sum = 0.0
            csi_sum = 0.0
            n_layers = 0
            for layer_name in layers:
                layer_sel = ckpt_sel.get(layer_name, {})
                if isinstance(layer_sel, dict):
                    ssi_sum += layer_sel.get('mean_ssi', 0.0)
                    csi_sum += layer_sel.get('mean_csi', 0.0)
                    n_layers += 1
            if n_layers > 0:
                mean_ssi_per_ckpt[ckpt_label] = ssi_sum / n_layers
                mean_csi_per_ckpt[ckpt_label] = csi_sum / n_layers

        # Key features: top CSI features at terminal checkpoint
        key_features: list[KeyFeatureInfo] = []
        terminal_ckpt = checkpoint_labels[-1] if checkpoint_labels else ''
        if terminal_ckpt:
            terminal_sel = selectivity.get(terminal_ckpt, {})
            for layer_name in layers:
                layer_sel = terminal_sel.get(layer_name, {})
                if not isinstance(layer_sel, dict):
                    continue
                csi_list = layer_sel.get('feature_csi', [])
                ssi_list = layer_sel.get('feature_ssi', [])
                for fi, csi_val in enumerate(csi_list):
                    if csi_val > 0.4:
                        ssi_val = ssi_list[fi] if fi < len(ssi_list) else 0.0
                        process = 'Di-H' if csi_val > 0.4 else ('Ab-H' if ssi_val > 0.3 else '')
                        key_features.append(KeyFeatureInfo(
                            feature_idx=fi,
                            layer_name=layer_name,
                            csi=csi_val,
                            ssi=ssi_val,
                            process=process,
                        ))
            key_features.sort(key=lambda f: f.csi, reverse=True)
            key_features = key_features[:10]

        # Sample event stats: per-process mean/std/min/max from transitions
        import numpy as np
        sample_event_stats: dict[str, SampleEventStats] = {}
        transitions_dir = _transitions_dir(sae_dir, matching_method)
        if transitions_dir.exists():
            per_sample_counts: dict[str, list[int]] = {'ab_h': [], 'tg_h': [], 'di_h': [], 'as_h': [], 'de_h': []}
            for pair_dir in sorted(transitions_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                for layer_file in pair_dir.glob('*.json'):
                    try:
                        import json
                        with open(layer_file) as f:
                            tr_data = json.load(f)
                        for event in tr_data:
                            if not isinstance(event, dict):
                                continue
                            if event.get('class_idx') != class_idx:
                                continue
                            pc = event.get('process_counts', {})
                            for proc in ['ab_h', 'tg_h', 'di_h', 'as_h', 'de_h']:
                                per_sample_counts[proc].append(pc.get(proc, 0))
                    except Exception:
                        continue
            for proc, counts in per_sample_counts.items():
                if counts:
                    arr = np.array(counts)
                    sample_event_stats[proc] = SampleEventStats(
                        mean=float(arr.mean()),
                        std=float(arr.std()),
                        min_val=int(arr.min()),
                        max_val=int(arr.max()),
                    )

        # Superclass name + sibling classes
        superclass_name = ''
        sibling_classes: list[str] = []
        superclass_summary = sae_data.get('superclass_summary', {})
        for sc_name, sc_data_raw in superclass_summary.items():
            if isinstance(sc_data_raw, dict):
                pcp = sc_data_raw.get('per_class_process', {})
                if class_name in pcp:
                    superclass_name = sc_name
                    sibling_classes = list(pcp.keys())
                    break

        # n_samples from activation data
        n_samples = 0
        n_checkpoints = len(checkpoint_labels)
        selected_classes = metadata.get('selected_classes', experiment.get('selected_classes', []))
        activations_dir = sae_dir / "activations"
        if activations_dir.exists() and checkpoint_labels and layers:
            first_act = activations_dir / checkpoint_labels[0] / f"{layers[0].replace('.', '_')}.npy"
            if first_act.exists():
                try:
                    H = np.load(str(first_act))
                    n_classes = max(len(selected_classes), 1)
                    n_samples = H.shape[0] // n_classes
                except Exception:
                    pass

        # Superclass Feature Participation: exclusive / partially shared / superclass-wide
        sc_participation = SuperclassParticipation(n_classes_in_superclass=len(sibling_classes))
        feature_directory_entries: list[FeatureDirectoryEntry] = []

        terminal_ckpt = checkpoint_labels[-1] if checkpoint_labels else ''
        if terminal_ckpt and activations_dir.exists() and sibling_classes and n_samples > 0:
            sibling_indices = [selected_classes.index(c) for c in sibling_classes if c in selected_classes]
            this_class_global_idx = selected_classes.index(class_name) if class_name in selected_classes else -1

            for layer_name in layers:
                safe_layer = layer_name.replace(".", "_")
                act_file = activations_dir / terminal_ckpt / f"{safe_layer}.npy"
                if not act_file.exists():
                    continue
                try:
                    H = np.load(str(act_file))
                    n_total = H.shape[0]
                    n_c = n_total // max(len(selected_classes), 1)
                    if n_c == 0:
                        continue

                    if this_class_global_idx < 0:
                        continue
                    this_rows = H[this_class_global_idx * n_c : (this_class_global_idx + 1) * n_c]
                    this_active = (np.abs(this_rows) > 1e-6)  # (n_c, d_hidden)

                    this_active_mask = this_active.any(axis=0)  # (d_hidden,)
                    this_feature_ids = np.where(this_active_mask)[0]

                    # Check sibling classes for each feature
                    for fid in this_feature_ids:
                        n_classes_active = 0
                        for ci in sibling_indices:
                            sibling_rows = H[ci * n_c : (ci + 1) * n_c]
                            if (np.abs(sibling_rows[:, fid]) > 1e-6).any():
                                n_classes_active += 1

                        if n_classes_active <= 1:
                            sc_participation.exclusive += 1
                            scope = "exclusive"
                        elif n_classes_active < len(sibling_indices):
                            sc_participation.partially_shared += 1
                            scope = "partial"
                        else:
                            sc_participation.superclass_wide += 1
                            scope = "superclass"

                        prevalence = float(this_active[:, fid].sum()) / n_c
                        active_vals = this_rows[this_active[:, fid], fid]
                        mean_act = float(active_vals.mean()) if len(active_vals) > 0 else 0.0

                        terminal_sel = selectivity.get(terminal_ckpt, {})
                        layer_sel = terminal_sel.get(layer_name, {})
                        ssi_list = layer_sel.get('feature_ssi', []) if isinstance(layer_sel, dict) else []
                        csi_list = layer_sel.get('feature_csi', []) if isinstance(layer_sel, dict) else []
                        ssi_val = ssi_list[fid] if fid < len(ssi_list) else 0.0
                        csi_val = csi_list[fid] if fid < len(csi_list) else 0.0

                        process = ''
                        if csi_val > 0.4:
                            process = 'di_h'
                        elif ssi_val > 0.3:
                            process = 'ab_h'

                        # Born at the first checkpoint where the feature is alive
                        born_at = terminal_ckpt
                        for ckpt_label in checkpoint_labels:
                            ckpt_file = activations_dir / ckpt_label / f"{safe_layer}.npy"
                            if ckpt_file.exists():
                                try:
                                    H_ckpt = np.load(str(ckpt_file))
                                    ckpt_rows = H_ckpt[this_class_global_idx * n_c : (this_class_global_idx + 1) * n_c]
                                    if (np.abs(ckpt_rows[:, fid]) > 1e-6).any():
                                        born_at = ckpt_label
                                        break
                                except Exception:
                                    pass

                        feature_directory_entries.append(FeatureDirectoryEntry(
                            layer_name=layer_name,
                            feature_idx=int(fid),
                            prevalence=round(prevalence, 3),
                            mean_activation=round(mean_act, 4),
                            born_at=born_at,
                            ssi=round(float(ssi_val), 4),
                            csi=round(float(csi_val), 4),
                            process=process,
                            scope=scope,
                        ))
                except Exception as e:
                    logger.warning(f"Could not compute feature directory for {layer_name}: {e}")

        # Sort feature directory: by layer depth (descending), then by prevalence descending
        def _layer_depth(name: str) -> int:
            # CNN: layer1..layer5
            if name.startswith('layer') and name[5:].isdigit():
                return int(name[5:])
            # ViT: blocks.0, blocks.1, ...
            parts = name.rsplit('.', 1)
            if len(parts) == 2 and parts[1].isdigit():
                return int(parts[1])
            return 0
        feature_directory_entries.sort(key=lambda f: (-_layer_depth(f.layer_name), -f.prevalence))
        # At most 20 features per layer
        limited_directory: list[FeatureDirectoryEntry] = []
        per_layer_count: dict[str, int] = {}
        for entry in feature_directory_entries:
            cnt = per_layer_count.get(entry.layer_name, 0)
            if cnt < 20:
                limited_directory.append(entry)
                per_layer_count[entry.layer_name] = cnt + 1
        feature_directory_entries = limited_directory

        return ClassViewData(
            class_idx=class_idx,
            class_name=class_name,
            process_summary=process_summary,
            sample_consistency=sample_consistency,
            mean_ssi_per_checkpoint=mean_ssi_per_ckpt,
            mean_csi_per_checkpoint=mean_csi_per_ckpt,
            key_features=key_features,
            sample_event_stats=sample_event_stats,
            superclass_name=superclass_name,
            n_samples=n_samples,
            n_checkpoints=n_checkpoints,
            superclass_participation=sc_participation,
            feature_directory=feature_directory_entries,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting class view: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Sample View

@router.get("/sample-detail/{stove_id}/{class_idx}/{sample_idx}", response_model=SampleDetailResponse)
async def get_sample_detail(stove_id: str, class_idx: int, sample_idx: int, matching_method: str = Query("activation", pattern="^(activation|weight)$")):
    """Get per-sample SAE encoding across all checkpoints + transitions."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)
        experiment_id = experiment['id']
        lane_id = lane['id']

        class_name = ""
        if class_idx < len(experiment['selected_classes']):
            class_name = experiment['selected_classes'][class_idx]

        checkpoints: list[SampleCheckpointData] = []
        transitions: list[SampleTransitionData] = []
        process_counts = {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0, "unclassified": 0, "total": 0}

        checkpoint_labels: list[str] = []
        layers: list[str] = []
        if sae_data:
            metadata = sae_data.get('metadata', {})
            layers = metadata.get('layers', [])
            checkpoint_labels = metadata.get('checkpoint_labels', [])

        # Build checkpoint data from per-sample activation files
        activations_dir = sae_dir / "activations"
        for ckpt_label in checkpoint_labels:
            per_layer_data: dict[str, dict[str, Any]] = {}
            for layer_name in layers:
                safe_layer = layer_name.replace(".", "_")
                act_file = activations_dir / ckpt_label / f"{safe_layer}.npy"
                if not act_file.exists():
                    continue
                try:
                    import numpy as np
                    H = np.load(str(act_file))  # (n_samples, d_hidden)
                    # Rows are ordered by sorted class_idx, with an equal number of samples per class
                    selected_classes = metadata.get('selected_classes', experiment.get('selected_classes', []))
                    n_samples_per_class = H.shape[0] // max(len(selected_classes), 1) if len(selected_classes) > 0 else 0
                    if n_samples_per_class > 0 and sample_idx < n_samples_per_class:
                        row_idx = class_idx * n_samples_per_class + sample_idx
                        if row_idx < H.shape[0]:
                            row = H[row_idx]
                            nonzero_mask = np.abs(row) > 1e-8
                            indices = np.where(nonzero_mask)[0].tolist()
                            values = row[nonzero_mask].tolist()

                            ssi_values: list[float] = []
                            csi_values: list[float] = []
                            if sae_data:
                                sel = sae_data.get('selectivity', {})
                                ckpt_sel = sel.get(ckpt_label, {})
                                layer_sel = ckpt_sel.get(layer_name, {})
                                feature_ssi_list = layer_sel.get('feature_ssi', [])
                                feature_csi_list = layer_sel.get('feature_csi', [])
                                for fi in indices:
                                    ssi_values.append(feature_ssi_list[fi] if fi < len(feature_ssi_list) else 0.0)
                                    csi_values.append(feature_csi_list[fi] if fi < len(feature_csi_list) else 0.0)

                            # Compute per-feature prevalence within the class
                            prevalence_values: list[float] = []
                            class_start = class_idx * n_samples_per_class
                            class_end = class_start + n_samples_per_class
                            if class_end <= H.shape[0]:
                                class_block = H[class_start:class_end]  # (n_per_class, d_hidden)
                                for fi in indices:
                                    col = class_block[:, fi]
                                    prev = float((np.abs(col) > 1e-8).sum()) / n_samples_per_class
                                    prevalence_values.append(prev)
                            else:
                                prevalence_values = [1.0] * len(indices)

                            per_layer_data[layer_name] = {
                                'indices': indices,
                                'values': values,
                                'ssi_values': ssi_values,
                                'csi_values': csi_values,
                                'prevalence_values': prevalence_values,
                                'n_active': len(indices),
                            }
                except Exception as e:
                    logger.warning(f"Could not load activation data {act_file}: {e}")

            checkpoints.append(SampleCheckpointData(
                checkpoint_label=ckpt_label,
                per_layer=per_layer_data,
            ))

        # Load transitions from per-sample transition files
        transitions_dir = _transitions_dir(sae_dir, matching_method)
        for i in range(1, len(checkpoint_labels)):
            from_ckpt = checkpoint_labels[i - 1]
            to_ckpt = checkpoint_labels[i]
            pair_key = f"{from_ckpt}_to_{to_ckpt}"

            transition_per_layer: dict[str, dict[str, Any]] = {}
            for layer_name in layers:
                safe_layer = layer_name.replace(".", "_")
                trans_file = transitions_dir / pair_key / f"{safe_layer}.json"
                if not trans_file.exists():
                    continue
                try:
                    with open(trans_file, 'r') as f:
                        trans_data = json.load(f)
                    # trans_data is a list of per-sample events
                    for ev in trans_data:
                        if ev.get('class_idx') == class_idx and ev.get('sample_idx_in_class') == sample_idx:
                            transition_per_layer[layer_name] = {
                                'born': ev.get('born', []),
                                'died': ev.get('died', []),
                                'stable': ev.get('stable', []),
                                'process_counts': ev.get('process_counts', {}),
                            }
                            for proc in ['ab_h', 'tg_h', 'di_h', 'as_h', 'de_h', 'unclassified']:
                                process_counts[proc] += ev.get('process_counts', {}).get(proc, 0)
                            break
                except Exception as e:
                    logger.warning(f"Could not load transition data {trans_file}: {e}")

            transitions.append(SampleTransitionData(
                from_checkpoint=from_ckpt,
                to_checkpoint=to_ckpt,
                per_layer=transition_per_layer,
            ))

        # tg_h is a subcategory of ab_h and is not added to the total
        process_counts["total"] = sum(process_counts[p] for p in ['ab_h', 'di_h', 'as_h', 'de_h', 'unclassified'])

        # Load sample image
        image_data = ""
        try:
            image_data = await _load_sample_image_base64(experiment, class_idx, sample_idx)
        except Exception as e:
            logger.warning(f"Could not load sample image for class {class_idx} sample {sample_idx}: {e}")

        return SampleDetailResponse(
            class_idx=class_idx,
            sample_idx=sample_idx,
            class_name=class_name,
            image_data=image_data,
            checkpoints=checkpoints,
            transitions=transitions,
            process_summary=ProcessCounts(**process_counts),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sample detail: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sample-layer-activity/{stove_id}/{class_idx}/{sample_idx}/{checkpoint}", response_model=SampleLayerActivity)
async def get_sample_layer_activity(stove_id: str, class_idx: int, sample_idx: int, checkpoint: str, matching_method: str = Query("activation", pattern="^(activation|weight)$")):
    """Get per-layer activation details for a sample at one checkpoint."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)

        if not sae_data:
            return SampleLayerActivity(checkpoint_label=checkpoint)

        sae_data = _resolve_method_data(sae_data, matching_method)

        metadata = sae_data.get('metadata', {})
        layers = metadata.get('layers', [])
        selected_classes = metadata.get('selected_classes', experiment.get('selected_classes', []))

        layer_data: dict[str, dict[str, Any]] = {}
        activations_dir = sae_dir / "activations"

        for layer_name in layers:
            safe_layer = layer_name.replace(".", "_")
            act_file = activations_dir / checkpoint / f"{safe_layer}.npy"
            if not act_file.exists():
                continue
            try:
                import numpy as np
                H = np.load(str(act_file))
                n_samples_per_class = H.shape[0] // max(len(selected_classes), 1) if len(selected_classes) > 0 else 0
                if n_samples_per_class > 0 and sample_idx < n_samples_per_class:
                    row_idx = class_idx * n_samples_per_class + sample_idx
                    if row_idx < H.shape[0]:
                        row = H[row_idx]
                        nonzero_mask = np.abs(row) > 1e-8
                        indices = np.where(nonzero_mask)[0].tolist()
                        values = row[nonzero_mask].tolist()

                        sel = sae_data.get('selectivity', {}).get(checkpoint, {}).get(layer_name, {})
                        ssi_list = sel.get('feature_ssi', [])
                        csi_list = sel.get('feature_csi', [])

                        features = []
                        for fi, val in zip(indices, values):
                            features.append({
                                'index': fi,
                                'value': val,
                                'ssi': ssi_list[fi] if fi < len(ssi_list) else 0.0,
                                'csi': csi_list[fi] if fi < len(csi_list) else 0.0,
                            })

                        layer_data[layer_name] = {
                            'n_active': len(indices),
                            'features': features,
                        }
            except Exception as e:
                logger.warning(f"Could not load layer activity {act_file}: {e}")

        return SampleLayerActivity(
            checkpoint_label=checkpoint,
            layers=layer_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting sample layer activity: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/sample-feature-flow/{stove_id}/{class_idx}/{sample_idx}/{from_ckpt}/{to_ckpt}/{layer}", response_model=FeatureFlowData)
async def get_sample_feature_flow(stove_id: str, class_idx: int, sample_idx: int, from_ckpt: str, to_ckpt: str, layer: str, matching_method: str = Query("activation", pattern="^(activation|weight)$")):
    """Get expanded feature flow between two checkpoints for a sample at a specific layer."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)

        pair_key = f"{from_ckpt}_to_{to_ckpt}"
        safe_layer = layer.replace(".", "_")
        trans_file = _transitions_dir(sae_dir, matching_method) / pair_key / f"{safe_layer}.json"

        if not trans_file.exists():
            return FeatureFlowData(from_checkpoint=from_ckpt, to_checkpoint=to_ckpt, layer_name=layer)

        with open(trans_file, 'r') as f:
            trans_data = json.load(f)

        for ev in trans_data:
            if ev.get('class_idx') == class_idx and ev.get('sample_idx_in_class') == sample_idx:
                return FeatureFlowData(
                    from_checkpoint=from_ckpt,
                    to_checkpoint=to_ckpt,
                    layer_name=layer,
                    born=ev.get('born', []),
                    died=ev.get('died', []),
                    stable=ev.get('stable', []),
                    process_labels=ev.get('process_counts', {}),
                )

        return FeatureFlowData(from_checkpoint=from_ckpt, to_checkpoint=to_ckpt, layer_name=layer)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature flow: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Summary View

@router.get("/summary/{stove_id}", response_model=SummaryViewData)
async def get_summary(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                      exclude_stable: bool = Query(False, description="Exclude stable features from process counts")):
    """Get data for the Summary View tab: hypotheses, null baseline, selectivity evolution."""
    try:
        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)

        if not sae_data:
            return SummaryViewData()

        metadata = sae_data.get('metadata', {})

        # Hypotheses
        hypotheses: dict[str, HypothesisVerdict] = {}
        for hyp_id, hyp_data in sae_data.get('hypotheses', {}).items():
            if not isinstance(hyp_data, dict):
                continue
            hypotheses[hyp_id] = HypothesisVerdict(
                hypothesis_id=hyp_id,
                verdict=hyp_data.get('verdict', 'insufficient_data'),
                events=hyp_data.get('events', 0),
                events_pct=hyp_data.get('events_pct', 0.0),
                evidence_met=hyp_data.get('evidence_met', 0),
                evidence_total=hyp_data.get('evidence_total', 0),
                evidence=hyp_data.get('evidence', []),
            )

        # Null baseline
        null_raw = sae_data.get('null_baseline', {})
        null_baseline: list[NullBaselinePoint] = []
        for pt in null_raw.get('per_checkpoint', []):
            if isinstance(pt, dict):
                null_baseline.append(NullBaselinePoint(
                    checkpoint=pt.get('checkpoint', ''),
                    observed_ssi=pt.get('observed_ssi', 0.0),
                    null_ssi=pt.get('null_ssi', 0.0),
                    p_value=pt.get('p_value', 1.0),
                ))

        # Selectivity evolution
        selectivity_evolution: list[SelectivityEvolutionPoint] = []
        for pt in sae_data.get('selectivity_evolution', []):
            if isinstance(pt, dict):
                selectivity_evolution.append(SelectivityEvolutionPoint(
                    checkpoint=pt.get('checkpoint', ''),
                    mean_ssi=pt.get('mean_ssi', 0.0),
                    mean_csi=pt.get('mean_csi', 0.0),
                    mean_sai=pt.get('mean_sai', 0.0),
                    n_alive_mean=pt.get('n_alive_mean', 0.0),
                ))

        # Process intensity (with optional stable exclusion)
        process_intensity_raw = sae_data.get('process_intensity', [])
        if exclude_stable:
            layers = metadata.get('layers', [])
            checkpoint_labels = metadata.get('checkpoint_labels', [])
            selectivity = sae_data_raw.get('selectivity', sae_data.get('selectivity', {}))
            adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
            adj_global = adjs['global']
            adjusted_pi = []
            for pi in process_intensity_raw:
                if not isinstance(pi, dict):
                    adjusted_pi.append(pi)
                    continue
                t = pi.get('transition', '')
                adj = adj_global.get(t, {})
                ab_h = max(0, pi.get('ab_h', 0) - adj.get('ab_h', 0))
                di_h = max(0, pi.get('di_h', 0) - adj.get('di_h', 0))
                total = ab_h + di_h + pi.get('as_h', 0) + pi.get('de_h', 0) + pi.get('unclassified', 0)
                adjusted_pi.append({**pi, 'ab_h': ab_h, 'di_h': di_h, 'total': total})
            process_intensity_raw = adjusted_pi

        # Within-checkpoint control (shared across methods, lives at top level)
        control_raw = sae_data_raw.get('within_checkpoint_control') if sae_data_raw else None
        control = None
        if control_raw and isinstance(control_raw, dict):
            per_ckpt = []
            for ckpt_data in control_raw.get('per_checkpoint', []):
                if isinstance(ckpt_data, dict):
                    layer_results = []
                    for lr in ckpt_data.get('per_layer', []):
                        if isinstance(lr, dict):
                            layer_results.append(ControlLayerResult(**lr))
                    per_ckpt.append(ControlCheckpointResult(
                        checkpoint=ckpt_data.get('checkpoint', ''),
                        per_layer=layer_results,
                    ))
            control = WithinCheckpointControl(
                per_checkpoint=per_ckpt,
                summary=control_raw.get('summary', {}),
            )

        return SummaryViewData(
            hypotheses=hypotheses,
            null_baseline=null_baseline,
            n_permutations=null_raw.get('n_permutations', 0),
            selectivity_evolution=selectivity_evolution,
            discrimination_gradients=sae_data.get('discrimination_gradients', {}),
            experiment_overview={
                'n_classes': metadata.get('n_classes', 0),
                'n_layers': metadata.get('n_layers', 0),
                'n_checkpoints': len(metadata.get('checkpoint_labels', [])),
                'approach': metadata.get('approach', ''),
                'expansion_factor': metadata.get('expansion_factor', 4),
                'k_sparse': metadata.get('k_sparse', 32),
            },
            process_intensity=process_intensity_raw,
            reconstruction_quality=sae_data.get('reconstruction_quality', {}),
            within_checkpoint_control=control,
            analysis_thresholds=metadata.get('analysis_thresholds'),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting summary: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Terminal Similarity

@router.get("/sample-terminal-similarity/{stove_id}/{class_idx}/{sample_idx}", response_model=SampleTerminalSimilarityResponse)
async def get_sample_terminal_similarity(stove_id: str, class_idx: int, sample_idx: int):
    """Compute cosine similarity of sample activations at each checkpoint to its terminal representation."""
    try:
        import numpy as np

        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)

        if not sae_data:
            return SampleTerminalSimilarityResponse(class_idx=class_idx, sample_idx=sample_idx)

        metadata = sae_data.get('metadata', {})
        layers = metadata.get('layers', [])
        checkpoint_labels = metadata.get('checkpoint_labels', [])
        selected_classes = metadata.get('selected_classes', experiment.get('selected_classes', []))

        if len(checkpoint_labels) < 2:
            return SampleTerminalSimilarityResponse(class_idx=class_idx, sample_idx=sample_idx)

        terminal_ckpt = checkpoint_labels[-1]
        activations_dir = sae_dir / "activations"
        similarities: list[TerminalSimilarityPoint] = []

        for layer_name in layers:
            safe_layer = layer_name.replace(".", "_")

            # Load terminal checkpoint activations for this layer
            terminal_file = activations_dir / terminal_ckpt / f"{safe_layer}.npy"
            if not terminal_file.exists():
                continue

            H_terminal = np.load(str(terminal_file))
            n_samples_per_class = H_terminal.shape[0] // max(len(selected_classes), 1) if len(selected_classes) > 0 else 0
            if n_samples_per_class == 0 or sample_idx >= n_samples_per_class:
                continue

            row_idx = class_idx * n_samples_per_class + sample_idx
            if row_idx >= H_terminal.shape[0]:
                continue

            terminal_vec = H_terminal[row_idx].astype(np.float64)
            terminal_norm = np.linalg.norm(terminal_vec)
            if terminal_norm < 1e-10:
                continue

            for ckpt_label in checkpoint_labels:
                act_file = activations_dir / ckpt_label / f"{safe_layer}.npy"
                if not act_file.exists():
                    continue

                H = np.load(str(act_file))
                if row_idx >= H.shape[0]:
                    continue

                ckpt_vec = H[row_idx].astype(np.float64)
                ckpt_norm = np.linalg.norm(ckpt_vec)
                if ckpt_norm < 1e-10:
                    cos_sim = 0.0
                else:
                    cos_sim = float(np.dot(ckpt_vec, terminal_vec) / (ckpt_norm * terminal_norm))

                similarities.append(TerminalSimilarityPoint(
                    checkpoint=ckpt_label,
                    layer_name=layer_name,
                    cosine_similarity=cos_sim,
                ))

        return SampleTerminalSimilarityResponse(
            class_idx=class_idx,
            sample_idx=sample_idx,
            similarities=similarities,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing terminal similarity: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Sample Image

@router.get("/sample-image/{stove_id}/{class_idx}/{sample_idx}", response_model=SampleImageResponse)
async def get_sample_image(stove_id: str, class_idx: int, sample_idx: int):
    """Load and return a sample image from the validation dataset."""
    try:
        stove, lane, experiment, sae_dir, sae_data = await _load_sae_context(stove_id)

        class_name = ""
        if class_idx < len(experiment['selected_classes']):
            class_name = experiment['selected_classes'][class_idx]

        image_data = await _load_sample_image_base64(experiment, class_idx, sample_idx)

        return SampleImageResponse(
            image_data=image_data,
            class_idx=class_idx,
            sample_idx=sample_idx,
            class_name=class_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading sample image: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Feature Catalog

@router.get("/feature-catalog/{stove_id}", response_model=FeatureCatalogResponse)
async def get_feature_catalog(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$")):
    """
    Global feature catalog: every alive feature at every checkpoint x layer,
    with selectivity metrics, matching state, and process labels.
    """
    try:
        import numpy as np

        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)
        if not sae_data:
            return FeatureCatalogResponse()

        metadata = sae_data.get('metadata', {})
        checkpoint_labels = metadata.get('checkpoint_labels', [])
        layers = metadata.get('layers', [])
        selectivity = sae_data_raw.get('selectivity', {})
        feature_matching = sae_data.get('feature_matching', {})
        activations_dir = sae_dir / "activations"

        # Build class -> superclass mapping for activation mass aggregation
        selected_classes = metadata.get('selected_classes', experiment.get('selected_classes', []))
        if isinstance(selected_classes, str):
            try:
                selected_classes = json.loads(selected_classes)
            except Exception:
                selected_classes = []
        sc_summary = sae_data.get('superclass_summary', {})
        superclass_names = sorted(sc_summary.keys())
        class_to_superclass: dict[str, str] = {}
        for sc_name, sc_data_item in sc_summary.items():
            if isinstance(sc_data_item, dict):
                for cname in sc_data_item.get('per_class_process', {}):
                    class_to_superclass[cname] = sc_name
        idx_to_classname: dict[int, str] = {i: name for i, name in enumerate(selected_classes)}
        idx_to_superclass: dict[int, str] = {}
        for i, cname in enumerate(selected_classes):
            if cname in class_to_superclass:
                idx_to_superclass[i] = class_to_superclass[cname]

        entries: list[CatalogFeatureEntry] = []

        # Build matching lookup: for each transition, which features are stable/born/died/transformed
        # Key: (checkpoint_label, layer_name) -> {fid: {status, matched_from, corr}}
        match_lookup: dict[tuple[str, str], dict[int, dict]] = {}

        for pair_key, layer_dict in feature_matching.items():
            parts = pair_key.split("->")
            if len(parts) != 2:
                continue
            ckpt_a, ckpt_b = parts
            for layer_name, match_info in layer_dict.items():
                if not isinstance(match_info, dict):
                    continue
                layer_lookup: dict[int, dict] = {}

                for s in match_info.get('stable', []):
                    if isinstance(s, (list, tuple)) and len(s) >= 3:
                        idx_a, idx_b, corr = int(s[0]), int(s[1]), float(s[2])
                        layer_lookup[idx_b] = {"status": "stable", "matched_from": idx_a, "corr": corr}

                for fid in match_info.get('born', []):
                    layer_lookup[int(fid)] = {"status": "born", "matched_from": None, "corr": None}

                for t in match_info.get('transformed', []):
                    if isinstance(t, (list, tuple)) and len(t) >= 3:
                        idx_a, idx_b, corr = int(t[0]), int(t[1]), float(t[2])
                        layer_lookup[idx_b] = {"status": "transformed", "matched_from": idx_a, "corr": corr}

                match_lookup[(ckpt_b, layer_name)] = layer_lookup

                # Died at ckpt_a, unless another transition already set the status
                died_lookup = match_lookup.setdefault((ckpt_a, layer_name), {})
                for fid in match_info.get('died', []):
                    if int(fid) not in died_lookup:
                        died_lookup[int(fid)] = {"status": "died_next", "matched_from": None, "corr": None}

        for ckpt_label in checkpoint_labels:
            ckpt_sel = selectivity.get(ckpt_label, {})

            for layer_name in layers:
                layer_sel = ckpt_sel.get(layer_name, {})
                if not isinstance(layer_sel, dict):
                    continue

                ssi_list = layer_sel.get('feature_ssi', [])
                csi_list = layer_sel.get('feature_csi', [])
                entropy_list = layer_sel.get('feature_entropy', [])
                fbi_list = layer_sel.get('feature_sai', [])
                n_active = layer_sel.get('n_active_features', 0)

                safe_layer = layer_name.replace(".", "_")
                act_file = activations_dir / ckpt_label / f"{safe_layer}.npy"
                H = None
                if act_file.exists():
                    try:
                        H = np.load(str(act_file))
                    except Exception:
                        pass

                n_features = len(ssi_list)
                n_samples_total = H.shape[0] if H is not None else 0

                layer_match = match_lookup.get((ckpt_label, layer_name), {})

                for fid in range(n_features):
                    ssi_val = float(ssi_list[fid]) if fid < len(ssi_list) else 0.0
                    csi_val = float(csi_list[fid]) if fid < len(csi_list) else 0.0
                    entropy_val = float(entropy_list[fid]) if fid < len(entropy_list) else 0.0
                    fbi_val = float(fbi_list[fid]) if fid < len(fbi_list) else 0.0

                    act_mean = 0.0
                    act_std = 0.0
                    n_active_samples = 0
                    act_rate = 0.0

                    if H is not None and fid < H.shape[1]:
                        col = H[:, fid]
                        active_mask = np.abs(col) > 1e-6
                        n_active_samples = int(active_mask.sum())
                        act_rate = n_active_samples / max(n_samples_total, 1)
                        if n_active_samples > 0:
                            active_vals = col[active_mask]
                            act_mean = float(active_vals.mean())
                            act_std = float(active_vals.std()) if n_active_samples > 1 else 0.0

                    # Skip dead features (inactive for all samples)
                    alive_thresh = max(1, n_samples_total * 0.01)
                    is_alive = n_active_samples >= alive_thresh
                    if not is_alive:
                        continue

                    match_info_entry = layer_match.get(fid, {})
                    match_status = match_info_entry.get("status", "")
                    matched_from = match_info_entry.get("matched_from")
                    match_corr = match_info_entry.get("corr")

                    processes: list[str] = []
                    if match_status == "born":
                        processes.append("as_h")
                    if ssi_val > 0.3:
                        processes.append("ab_h")
                    if csi_val > 0.4:
                        processes.append("di_h")
                    if match_status == "died_next":
                        processes.append("de_h")

                    # Activation mass: per-class and per-superclass active sample counts
                    act_mass_data = layer_sel.get('activation_mass', {})
                    mass_by_class: dict[str, int] = {}
                    mass_by_sc: dict[str, int] = {}
                    for ci_str, counts in act_mass_data.items():
                        ci = int(ci_str)
                        if fid < len(counts):
                            count_val = int(counts[fid])
                            cname = idx_to_classname.get(ci, '')
                            if cname:
                                mass_by_class[cname] = count_val
                            sc_name = idx_to_superclass.get(ci, '')
                            if sc_name:
                                mass_by_sc[sc_name] = mass_by_sc.get(sc_name, 0) + count_val
                    best_sc = max(mass_by_sc, key=mass_by_sc.get) if mass_by_sc else ''

                    entries.append(CatalogFeatureEntry(
                        feature_idx=fid,
                        layer_name=layer_name,
                        checkpoint=ckpt_label,
                        alive=is_alive,
                        activation_mean=round(act_mean, 6),
                        activation_std=round(act_std, 6),
                        n_samples_active=n_active_samples,
                        n_samples_total=n_samples_total,
                        activation_rate=round(act_rate, 4),
                        ssi=round(ssi_val, 4),
                        csi=round(csi_val, 4),
                        entropy=round(entropy_val, 4),
                        sai=round(fbi_val, 4),
                        match_status=match_status if match_status != "died_next" else "",
                        matched_from_idx=matched_from,
                        match_correlation=round(match_corr, 4) if match_corr is not None else None,
                        processes=processes,
                        activation_mass_by_superclass=mass_by_sc,
                        activation_mass_by_class=mass_by_class,
                        best_superclass=best_sc,
                    ))

        return FeatureCatalogResponse(
            checkpoint_labels=checkpoint_labels,
            layers=layers,
            n_total_features=len(entries),
            features=entries,
            superclass_names=superclass_names,
            class_names=[str(c) for c in selected_classes],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature catalog: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Process Catalog

@router.get("/process-catalog/{stove_id}", response_model=ProcessCatalogResponse)
async def get_process_catalog(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$")):
    """
    Process event catalog: every unique feature-level event (born/died/stable)
    at each transition, aggregated across samples from transition JSON files.
    """
    try:
        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)
        if not sae_data:
            return ProcessCatalogResponse()

        metadata = sae_data.get('metadata', {})
        checkpoint_labels = metadata.get('checkpoint_labels', [])
        layers = metadata.get('layers', [])
        selectivity = sae_data_raw.get('selectivity', {})

        transitions_subdir = "transitions" if matching_method == "activation" else "transitions_weight"
        transitions_dir = sae_dir / transitions_subdir

        # Build transition pairs from checkpoint labels
        transitions_list: list[str] = []
        for i in range(len(checkpoint_labels) - 1):
            transitions_list.append(f"{checkpoint_labels[i]}->{checkpoint_labels[i+1]}")

        entries: list[ProcessCatalogEntry] = []

        for pair_key in transitions_list:
            parts = pair_key.split("->")
            ckpt_a, ckpt_b = parts[0], parts[1]
            safe_pair = pair_key.replace("->", "_to_")

            for layer_name in layers:
                safe_layer = layer_name.replace(".", "_")
                trans_file = transitions_dir / safe_pair / f"{safe_layer}.json"
                if not trans_file.exists():
                    continue

                try:
                    with open(trans_file) as f:
                        sample_events = json.load(f)
                except Exception:
                    continue

                n_samples_total = len(sample_events)

                # Aggregate per-feature events across all samples
                # born: fid -> {count, processes (union), ssi_sum, csi_sum}
                born_agg: dict[int, dict] = {}
                died_agg: dict[int, dict] = {}
                stable_agg: dict[int, dict] = {}  # keyed by fid_b

                for ev in sample_events:
                    for b in ev.get('born', []):
                        fid = b['fid']
                        if fid not in born_agg:
                            born_agg[fid] = {"count": 0, "processes": set(), "ssi_sum": 0.0, "csi_sum": 0.0}
                        born_agg[fid]["count"] += 1
                        for p in b.get('processes', []):
                            born_agg[fid]["processes"].add(p)
                        born_agg[fid]["ssi_sum"] += b.get('ssi', 0.0)
                        born_agg[fid]["csi_sum"] += b.get('csi', 0.0)

                    for d in ev.get('died', []):
                        fid = d['fid']
                        if fid not in died_agg:
                            died_agg[fid] = {"count": 0, "processes": set()}
                        died_agg[fid]["count"] += 1
                        for p in d.get('processes', []):
                            died_agg[fid]["processes"].add(p)

                    for s in ev.get('stable', []):
                        fid_b = s['fid_b']
                        fid_a = s['fid_a']
                        if fid_b not in stable_agg:
                            stable_agg[fid_b] = {"count": 0, "fid_a": fid_a}
                        stable_agg[fid_b]["count"] += 1

                ckpt_sel_b = selectivity.get(ckpt_b, {}).get(layer_name, {})
                ssi_list_b = ckpt_sel_b.get('feature_ssi', [])
                csi_list_b = ckpt_sel_b.get('feature_csi', [])
                ckpt_sel_a = selectivity.get(ckpt_a, {}).get(layer_name, {})
                ssi_list_a = ckpt_sel_a.get('feature_ssi', [])
                csi_list_a = ckpt_sel_a.get('feature_csi', [])

                # Emit born entries
                for fid, agg in born_agg.items():
                    ssi_val = float(ssi_list_b[fid]) if fid < len(ssi_list_b) else (agg["ssi_sum"] / max(agg["count"], 1))
                    csi_val = float(csi_list_b[fid]) if fid < len(csi_list_b) else (agg["csi_sum"] / max(agg["count"], 1))
                    entries.append(ProcessCatalogEntry(
                        feature_idx=fid,
                        layer_name=layer_name,
                        transition=pair_key,
                        event_type="born",
                        processes=sorted(agg["processes"]),
                        n_samples_active=agg["count"],
                        n_samples_total=n_samples_total,
                        prevalence=round(agg["count"] / max(n_samples_total, 1), 4),
                        ssi=round(ssi_val, 4),
                        csi=round(csi_val, 4),
                    ))

                # Emit died entries
                for fid, agg in died_agg.items():
                    ssi_val = float(ssi_list_a[fid]) if fid < len(ssi_list_a) else 0.0
                    csi_val = float(csi_list_a[fid]) if fid < len(csi_list_a) else 0.0
                    entries.append(ProcessCatalogEntry(
                        feature_idx=fid,
                        layer_name=layer_name,
                        transition=pair_key,
                        event_type="died",
                        processes=sorted(agg["processes"]),
                        n_samples_active=agg["count"],
                        n_samples_total=n_samples_total,
                        prevalence=round(agg["count"] / max(n_samples_total, 1), 4),
                        ssi=round(ssi_val, 4),
                        csi=round(csi_val, 4),
                    ))

                # Emit stable entries
                for fid_b, agg in stable_agg.items():
                    ssi_val = float(ssi_list_b[fid_b]) if fid_b < len(ssi_list_b) else 0.0
                    csi_val = float(csi_list_b[fid_b]) if fid_b < len(csi_list_b) else 0.0
                    processes: list[str] = []
                    if ssi_val > 0.3:
                        processes.append("ab_h")
                    if csi_val > 0.4:
                        processes.append("di_h")
                    entries.append(ProcessCatalogEntry(
                        feature_idx=fid_b,
                        layer_name=layer_name,
                        transition=pair_key,
                        event_type="stable",
                        processes=processes,
                        n_samples_active=agg["count"],
                        n_samples_total=n_samples_total,
                        prevalence=round(agg["count"] / max(n_samples_total, 1), 4),
                        ssi=round(ssi_val, 4),
                        csi=round(csi_val, 4),
                        matched_from_idx=agg["fid_a"],
                    ))

        return ProcessCatalogResponse(
            transitions=transitions_list,
            layers=layers,
            n_total_events=len(entries),
            events=entries,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting process catalog: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Granular Process Catalog

@router.get("/granular-process-catalog/{stove_id}", response_model=GranularProcessCatalogResponse)
async def get_granular_process_catalog(
    stove_id: str,
    matching_method: str = Query("activation", pattern="^(activation|weight)$"),
    layer: str = Query(..., description="Layer name (required; one layer at a time)"),
    transition: str = Query(None, description="Transition filter, e.g. '0->1'"),
    event_type: str = Query(None, description="Filter: born, died, stable"),
    process: str = Query(None, description="Filter: ab_h, tg_h, di_h, as_h, de_h"),
    class_idx: int = Query(None, description="Filter by class index"),
    sample_idx: int = Query(None, description="Filter by sample index within class"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    limit: int = Query(500, ge=1, le=5000, description="Page size"),
):
    """
    Granular (per-sample) process catalog: every atomic feature event
    for every sample, without aggregation. Requires a layer filter to
    keep response sizes manageable.
    """
    try:
        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)
        if not sae_data:
            return GranularProcessCatalogResponse()

        metadata = sae_data.get('metadata', {})
        checkpoint_labels = metadata.get('checkpoint_labels', [])
        layers = metadata.get('layers', [])
        selectivity = sae_data_raw.get('selectivity', {})

        selected_classes: list[str] = []
        try:
            selected_classes = json.loads(experiment['selected_classes'])
        except Exception:
            selected_classes = metadata.get('selected_classes', [])

        if layer not in layers:
            raise HTTPException(status_code=400, detail=f"Layer '{layer}' not found. Available: {layers}")

        transitions_subdir = "transitions" if matching_method == "activation" else "transitions_weight"
        transitions_dir = sae_dir / transitions_subdir

        transitions_list: list[str] = []
        for i in range(len(checkpoint_labels) - 1):
            transitions_list.append(f"{checkpoint_labels[i]}->{checkpoint_labels[i+1]}")

        target_transitions = transitions_list
        if transition:
            if transition not in transitions_list:
                raise HTTPException(status_code=400, detail=f"Transition '{transition}' not found. Available: {transitions_list}")
            target_transitions = [transition]

        all_entries: list[GranularProcessEntry] = []

        for pair_key in target_transitions:
            parts = pair_key.split("->")
            ckpt_a, ckpt_b = parts[0], parts[1]
            safe_pair = pair_key.replace("->", "_to_")
            safe_layer = layer.replace(".", "_")
            trans_file = transitions_dir / safe_pair / f"{safe_layer}.json"
            if not trans_file.exists():
                continue

            try:
                with open(trans_file) as f:
                    sample_events = json.load(f)
            except Exception:
                continue

            # Selectivity data for this checkpoint+layer
            ckpt_sel_b = selectivity.get(ckpt_b, {}).get(layer, {})
            ssi_list_b = ckpt_sel_b.get('feature_ssi', [])
            csi_list_b = ckpt_sel_b.get('feature_csi', [])
            ckpt_sel_a = selectivity.get(ckpt_a, {}).get(layer, {})
            ssi_list_a = ckpt_sel_a.get('feature_ssi', [])
            csi_list_a = ckpt_sel_a.get('feature_csi', [])

            for ev in sample_events:
                ev_class_idx = ev.get('class_idx', 0)
                ev_sample_idx = ev.get('sample_idx_in_class', 0)

                if class_idx is not None and ev_class_idx != class_idx:
                    continue
                if sample_idx is not None and ev_sample_idx != sample_idx:
                    continue

                class_name = selected_classes[ev_class_idx] if ev_class_idx < len(selected_classes) else ""

                # Born events
                if event_type is None or event_type == "born":
                    for b in ev.get('born', []):
                        fid = b['fid']
                        procs = b.get('processes', [])
                        if process and process not in procs:
                            continue
                        ssi_val = float(ssi_list_b[fid]) if fid < len(ssi_list_b) else b.get('ssi', 0.0)
                        csi_val = float(csi_list_b[fid]) if fid < len(csi_list_b) else b.get('csi', 0.0)
                        all_entries.append(GranularProcessEntry(
                            class_idx=ev_class_idx,
                            class_name=class_name,
                            sample_idx=ev_sample_idx,
                            feature_idx=fid,
                            layer_name=layer,
                            transition=pair_key,
                            event_type="born",
                            processes=sorted(procs),
                            ssi=round(ssi_val, 4),
                            csi=round(csi_val, 4),
                        ))

                # Died events
                if event_type is None or event_type == "died":
                    for d in ev.get('died', []):
                        fid = d['fid']
                        procs = d.get('processes', ['de_h'])
                        if process and process not in procs:
                            continue
                        ssi_val = float(ssi_list_a[fid]) if fid < len(ssi_list_a) else 0.0
                        csi_val = float(csi_list_a[fid]) if fid < len(csi_list_a) else 0.0
                        all_entries.append(GranularProcessEntry(
                            class_idx=ev_class_idx,
                            class_name=class_name,
                            sample_idx=ev_sample_idx,
                            feature_idx=fid,
                            layer_name=layer,
                            transition=pair_key,
                            event_type="died",
                            processes=sorted(procs),
                            ssi=round(ssi_val, 4),
                            csi=round(csi_val, 4),
                        ))

                # Stable events
                if event_type is None or event_type == "stable":
                    for s in ev.get('stable', []):
                        fid_b = s['fid_b']
                        fid_a = s['fid_a']
                        ssi_val = float(ssi_list_b[fid_b]) if fid_b < len(ssi_list_b) else 0.0
                        csi_val = float(csi_list_b[fid_b]) if fid_b < len(csi_list_b) else 0.0
                        procs: list[str] = []
                        if ssi_val > 0.3:
                            procs.append("ab_h")
                        if csi_val > 0.4:
                            procs.append("di_h")
                        if process and process not in procs:
                            continue
                        all_entries.append(GranularProcessEntry(
                            class_idx=ev_class_idx,
                            class_name=class_name,
                            sample_idx=ev_sample_idx,
                            feature_idx=fid_b,
                            layer_name=layer,
                            transition=pair_key,
                            event_type="stable",
                            processes=procs,
                            ssi=round(ssi_val, 4),
                            csi=round(csi_val, 4),
                            matched_from_idx=fid_a,
                        ))

        n_total = len(all_entries)
        page = all_entries[offset:offset + limit]

        return GranularProcessCatalogResponse(
            transitions=transitions_list,
            layers=layers,
            class_names=selected_classes,
            n_total_events=n_total,
            offset=offset,
            limit=limit,
            events=page,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting granular process catalog: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Insights helpers

import math as _math

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace inf/NaN floats with None so JSON serialization succeeds."""
    if isinstance(obj, float):
        if _math.isinf(obj) or _math.isnan(obj):
            return None
        return obj
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if _math.isinf(v) or _math.isnan(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


# Insights (Scientific Deep-Dive)

def _compute_boundary_crystallization(sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
                                       selected_classes: list[str], class_to_sc: dict[str, str],
                                       n_classes: int, k_sparse: int) -> BoundaryCrystallizationData:
    """Compute within vs between superclass feature overlap at each checkpoint."""
    import numpy as np

    n_samples_per_class = 100  # samples per class in the CIFAR-100 subsets
    points = []

    for ckpt in checkpoint_labels:
        for layer in layers:
            act_path = sae_dir / "activations" / ckpt / f"{_safe_layer_filename(layer)}.npy"
            if not act_path.exists():
                continue

            try:
                activations = np.load(str(act_path))  # (n_total_samples, d_hidden)
                n_total = activations.shape[0]
                actual_n_per_class = n_total // n_classes
                if actual_n_per_class == 0:
                    continue

                # Compute per-class active feature sets (binary: feature fires on >= 10% of class samples)
                class_feature_sets = []
                for ci in range(n_classes):
                    start = ci * actual_n_per_class
                    end = start + actual_n_per_class
                    class_acts = activations[start:end]  # (n_per_class, d_hidden)
                    # A feature is "class-active" if it fires (non-zero) on >= 10% of samples
                    firing_rate = (np.abs(class_acts) > 1e-8).mean(axis=0)  # (d_hidden,)
                    active_features = set(np.where(firing_rate >= 0.1)[0])
                    class_feature_sets.append(active_features)

                within_overlaps = []
                between_overlaps = []

                for i in range(n_classes):
                    sc_i = class_to_sc.get(selected_classes[i], "")
                    for j in range(i + 1, n_classes):
                        sc_j = class_to_sc.get(selected_classes[j], "")
                        set_i = class_feature_sets[i]
                        set_j = class_feature_sets[j]

                        if not set_i and not set_j:
                            jaccard = 0.0
                        else:
                            intersection = len(set_i & set_j)
                            union = len(set_i | set_j)
                            jaccard = intersection / union if union > 0 else 0.0

                        if sc_i == sc_j and sc_i:
                            within_overlaps.append(jaccard)
                        else:
                            between_overlaps.append(jaccard)

                within_mean = float(np.mean(within_overlaps)) if within_overlaps else 0.0
                between_mean = float(np.mean(between_overlaps)) if between_overlaps else 0.0
                divergence = within_mean - between_mean
                cryst_idx = (within_mean / between_mean) if between_mean > 0.01 else within_mean * 100

                points.append(BoundaryCrystallizationPoint(
                    checkpoint=ckpt,
                    layer=layer,
                    within_sc_overlap=round(within_mean, 4),
                    between_sc_overlap=round(between_mean, 4),
                    divergence=round(divergence, 4),
                    crystallization_index=round(min(cryst_idx, 10.0), 4),
                ))
            except Exception as e:
                logger.warning(f"Error computing crystallization for {ckpt}/{layer}: {e}")
                continue

    return BoundaryCrystallizationData(
        points=points,
        layers=layers,
        checkpoint_labels=checkpoint_labels,
    )


def _compute_temporal_ordering(sae_data: dict, sc_summary: dict,
                                selected_classes: list[str],
                                class_to_sc: dict[str, str],
                                adj_per_class: Optional[dict] = None) -> TemporalOrderingData:
    """Test whether Ab-H events precede Di-H events across superclasses.

    Tracks three abstraction views:
    - ab_h (combined): all abstraction events (SSI-driven + SAI-driven)
    - ab_h_specific: superclass-specific abstraction only (ab_h - tg_h)
    - tg_h: task-general abstraction only (SAI-driven)

    If adj_per_class is provided (from _compute_stable_adjustments), stable
    feature contributions are subtracted from Ab-H/Di-H counts.
    """
    class_process = sae_data.get('class_process_summary', {})

    # Aggregate per-superclass per-transition Ab-H, Tg-H and Di-H counts
    sc_names = sorted(sc_summary.keys())
    transitions = []

    # Transition list from the first class
    first_class = next(iter(class_process.values()), None)
    if first_class and 'per_transition' in first_class:
        transitions = [t['transition'] for t in first_class['per_transition']]

    entries = []
    onset_ab_h_list = []
    onset_di_h_list = []
    onset_ab_h_specific_list = []
    onset_tg_h_list = []

    for sc_name in sc_names:
        sc_classes = [cls for cls in selected_classes if class_to_sc.get(cls) == sc_name]
        sc_class_indices = [str(selected_classes.index(cls)) for cls in sc_classes]

        ab_h_per_trans = []
        tg_h_per_trans = []
        di_h_per_trans = []

        for t_idx, trans in enumerate(transitions):
            ab_h_sum = 0
            tg_h_sum = 0
            di_h_sum = 0
            for ci in sc_class_indices:
                cls_data = class_process.get(ci, {})
                per_trans = cls_data.get('per_transition', [])
                if t_idx < len(per_trans):
                    raw_ab = per_trans[t_idx].get('ab_h', 0)
                    raw_tg = per_trans[t_idx].get('tg_h', 0)
                    raw_di = per_trans[t_idx].get('di_h', 0)
                    # Subtract stable contributions if requested
                    if adj_per_class:
                        ci_int = int(ci) if isinstance(ci, str) else ci
                        t_adj = adj_per_class.get(ci_int, {}).get(trans, {})
                        raw_ab = max(0, raw_ab - t_adj.get('ab_h', 0))
                        raw_tg = max(0, raw_tg - t_adj.get('tg_h', 0))
                        raw_di = max(0, raw_di - t_adj.get('di_h', 0))
                    ab_h_sum += raw_ab
                    tg_h_sum += raw_tg
                    di_h_sum += raw_di
            ab_h_per_trans.append(ab_h_sum)
            tg_h_per_trans.append(tg_h_sum)
            di_h_per_trans.append(di_h_sum)

        # Derive superclass-specific = ab_h - tg_h
        ab_h_specific_per_trans = [max(0, a - t) for a, t in zip(ab_h_per_trans, tg_h_per_trans)]

        # Compute cumulative fractions for all four series
        total_ab = sum(ab_h_per_trans) or 1
        total_ab_spec = sum(ab_h_specific_per_trans) or 1
        total_tg = sum(tg_h_per_trans) or 1
        total_di = sum(di_h_per_trans) or 1

        ab_h_cumulative = []
        ab_h_specific_cumulative = []
        tg_h_cumulative = []
        di_h_cumulative = []
        cum_ab, cum_ab_spec, cum_tg, cum_di = 0, 0, 0, 0
        for t_idx in range(len(transitions)):
            cum_ab += ab_h_per_trans[t_idx]
            cum_ab_spec += ab_h_specific_per_trans[t_idx]
            cum_tg += tg_h_per_trans[t_idx]
            cum_di += di_h_per_trans[t_idx]
            ab_h_cumulative.append(round(cum_ab / total_ab, 4))
            ab_h_specific_cumulative.append(round(cum_ab_spec / total_ab_spec, 4))
            tg_h_cumulative.append(round(cum_tg / total_tg, 4))
            di_h_cumulative.append(round(cum_di / total_di, 4))

        # Find onset: first transition where cumulative >= 0.1 (10% threshold)
        threshold = 0.1
        ab_h_onset = -1
        ab_h_specific_onset = -1
        tg_h_onset = -1
        di_h_onset = -1
        for t_idx in range(len(transitions)):
            if ab_h_onset < 0 and ab_h_cumulative[t_idx] >= threshold:
                ab_h_onset = t_idx
            if ab_h_specific_onset < 0 and ab_h_specific_cumulative[t_idx] >= threshold:
                ab_h_specific_onset = t_idx
            if tg_h_onset < 0 and tg_h_cumulative[t_idx] >= threshold:
                tg_h_onset = t_idx
            if di_h_onset < 0 and di_h_cumulative[t_idx] >= threshold:
                di_h_onset = t_idx

        entries.append(TemporalOrderingEntry(
            superclass=sc_name,
            ab_h_onset_idx=ab_h_onset,
            di_h_onset_idx=di_h_onset,
            ab_h_cumulative=ab_h_cumulative,
            di_h_cumulative=di_h_cumulative,
            ab_h_specific_onset_idx=ab_h_specific_onset,
            tg_h_onset_idx=tg_h_onset,
            ab_h_specific_cumulative=ab_h_specific_cumulative,
            tg_h_cumulative=tg_h_cumulative,
        ))

        if ab_h_onset >= 0:
            onset_ab_h_list.append(ab_h_onset)
        if di_h_onset >= 0:
            onset_di_h_list.append(di_h_onset)
        if ab_h_specific_onset >= 0:
            onset_ab_h_specific_list.append(ab_h_specific_onset)
        if tg_h_onset >= 0:
            onset_tg_h_list.append(tg_h_onset)

    # Leads of combined ab_h vs di_h
    ab_leads = sum(1 for e in entries if 0 <= e.ab_h_onset_idx < e.di_h_onset_idx or (e.ab_h_onset_idx >= 0 and e.di_h_onset_idx < 0))
    di_leads = sum(1 for e in entries if 0 <= e.di_h_onset_idx < e.ab_h_onset_idx or (e.di_h_onset_idx >= 0 and e.ab_h_onset_idx < 0))
    tied = len(entries) - ab_leads - di_leads

    # Leads of ab_h_specific vs di_h
    ab_spec_leads = sum(1 for e in entries if 0 <= e.ab_h_specific_onset_idx < e.di_h_onset_idx or (e.ab_h_specific_onset_idx >= 0 and e.di_h_onset_idx < 0))
    # Leads of tg_h vs di_h
    tg_leads_di = sum(1 for e in entries if 0 <= e.tg_h_onset_idx < e.di_h_onset_idx or (e.tg_h_onset_idx >= 0 and e.di_h_onset_idx < 0))

    # Wilcoxon signed-rank test: combined ab_h vs di_h
    wilcoxon_p = None
    wilcoxon_specific_p = None
    try:
        from scipy.stats import wilcoxon
        paired = [(e.ab_h_onset_idx, e.di_h_onset_idx) for e in entries if e.ab_h_onset_idx >= 0 and e.di_h_onset_idx >= 0]
        if len(paired) >= 3:
            ab_vals = [p[0] for p in paired]
            di_vals = [p[1] for p in paired]
            if ab_vals != di_vals:
                stat, p_val = wilcoxon(ab_vals, di_vals)
                wilcoxon_p = round(float(p_val), 6)
        # Wilcoxon for ab_h_specific vs di_h
        paired_spec = [(e.ab_h_specific_onset_idx, e.di_h_onset_idx) for e in entries if e.ab_h_specific_onset_idx >= 0 and e.di_h_onset_idx >= 0]
        if len(paired_spec) >= 3:
            ab_spec_vals = [p[0] for p in paired_spec]
            di_spec_vals = [p[1] for p in paired_spec]
            if ab_spec_vals != di_spec_vals:
                stat, p_val = wilcoxon(ab_spec_vals, di_spec_vals)
                wilcoxon_specific_p = round(float(p_val), 6)
    except ImportError:
        pass

    mean_ab = float(np.mean(onset_ab_h_list)) if onset_ab_h_list else -1
    mean_di = float(np.mean(onset_di_h_list)) if onset_di_h_list else -1
    mean_ab_spec = float(np.mean(onset_ab_h_specific_list)) if onset_ab_h_specific_list else -1
    mean_tg = float(np.mean(onset_tg_h_list)) if onset_tg_h_list else -1

    return TemporalOrderingData(
        per_superclass=entries,
        transitions=transitions,
        wilcoxon_p=wilcoxon_p,
        mean_ab_h_onset=round(mean_ab, 2),
        mean_di_h_onset=round(mean_di, 2),
        ab_h_leads_count=ab_leads,
        di_h_leads_count=di_leads,
        tied_count=tied,
        wilcoxon_specific_p=wilcoxon_specific_p,
        mean_ab_h_specific_onset=round(mean_ab_spec, 2),
        mean_tg_h_onset=round(mean_tg, 2),
        ab_h_specific_leads_count=ab_spec_leads,
        tg_h_leads_di_h_count=tg_leads_di,
    )


def _compute_feature_genealogy(sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
                                sae_data: dict, matching_method: str = "activation") -> FeatureGenealogyData:
    """Build feature lineage trees across checkpoints."""
    import numpy as np

    all_lineages = []
    lineage_id_counter = 0

    # Build selectivity lookup: checkpoint -> layer -> {feature_ssi: [...], feature_csi: [...]}
    selectivity = sae_data.get('selectivity', {})

    for layer in layers:
        # Step 1: Build forward map from transition files
        # forward_map[ckpt_label] = {src_fid: (dst_fid, n_samples)}
        forward_maps = {}
        # Also track born features at each transition
        born_at = {}  # ckpt_label -> set of fids born at that checkpoint

        transitions_dir = _transitions_dir(sae_dir, matching_method)

        for t_idx in range(len(checkpoint_labels) - 1):
            ckpt_a = checkpoint_labels[t_idx]
            ckpt_b = checkpoint_labels[t_idx + 1]
            pair_key = f"{ckpt_a}_to_{ckpt_b}"

            trans_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not trans_file.exists():
                continue

            try:
                with open(trans_file, 'r') as f:
                    samples = json.load(f)
            except Exception:
                continue

            # Count stable links across all samples
            link_counts = {}  # (fid_a, fid_b) -> count
            born_counts = {}  # fid_b -> count

            for sample in samples:
                for stable_entry in sample.get('stable', []):
                    fid_a = stable_entry.get('fid_a')
                    fid_b = stable_entry.get('fid_b')
                    if fid_a is not None and fid_b is not None:
                        key = (fid_a, fid_b)
                        link_counts[key] = link_counts.get(key, 0) + 1

                for born_entry in sample.get('born', []):
                    fid = born_entry.get('fid')
                    if fid is not None:
                        born_counts[fid] = born_counts.get(fid, 0) + 1

            # For each source feature, find the strongest link (most samples)
            src_to_best = {}
            for (fid_a, fid_b), count in link_counts.items():
                if fid_a not in src_to_best or count > src_to_best[fid_a][1]:
                    src_to_best[fid_a] = (fid_b, count)

            forward_maps[ckpt_a] = src_to_best
            born_at[ckpt_b] = born_counts

        # Step 2: Get alive features at checkpoint 0
        landscape = sae_data.get('feature_landscape', {})
        first_ckpt = checkpoint_labels[0]
        layer_landscape = landscape.get(first_ckpt, {}).get(layer, {})
        n_alive_initial = layer_landscape.get('n_alive', 0)

        sel_0 = selectivity.get(first_ckpt, {}).get(layer, {})
        ssi_list_0 = sel_0.get('feature_ssi', [])
        csi_list_0 = sel_0.get('feature_csi', [])

        # Alive features: non-zero activity in the activation matrix
        act_path = sae_dir / "activations" / first_ckpt / f"{_safe_layer_filename(layer)}.npy"
        founder_fids = set()
        if act_path.exists():
            try:
                acts = np.load(str(act_path))
                firing_rate = (np.abs(acts) > 1e-8).mean(axis=0)
                founder_fids = set(np.where(firing_rate >= 0.01)[0])
            except Exception:
                founder_fids = set(range(n_alive_initial))
        else:
            founder_fids = set(range(n_alive_initial))

        # Step 3: Trace lineages from founders
        for founder_fid in founder_fids:
            nodes = []
            current_fid = founder_fid

            # Get SSI/CSI for founder
            founder_ssi = ssi_list_0[founder_fid] if founder_fid < len(ssi_list_0) else 0.0
            founder_csi = csi_list_0[founder_fid] if founder_fid < len(csi_list_0) else 0.0

            nodes.append(LineageNode(
                checkpoint=first_ckpt,
                feature_idx=current_fid,
                ssi=round(founder_ssi, 4),
                csi=round(founder_csi, 4),
                n_samples=0,
            ))

            for t_idx in range(len(checkpoint_labels) - 1):
                ckpt_a = checkpoint_labels[t_idx]
                fwd = forward_maps.get(ckpt_a, {})

                if current_fid in fwd:
                    next_fid, n_samples = fwd[current_fid]
                    ckpt_b = checkpoint_labels[t_idx + 1]

                    sel_b = selectivity.get(ckpt_b, {}).get(layer, {})
                    ssi_list_b = sel_b.get('feature_ssi', [])
                    csi_list_b = sel_b.get('feature_csi', [])
                    node_ssi = ssi_list_b[next_fid] if next_fid < len(ssi_list_b) else 0.0
                    node_csi = csi_list_b[next_fid] if next_fid < len(csi_list_b) else 0.0

                    nodes.append(LineageNode(
                        checkpoint=ckpt_b,
                        feature_idx=next_fid,
                        ssi=round(node_ssi, 4),
                        csi=round(node_csi, 4),
                        n_samples=n_samples,
                    ))
                    current_fid = next_fid
                else:
                    break  # lineage died

            depth = len(nodes)
            if depth >= 3:  # Only keep lineages spanning 3+ checkpoints
                max_ssi = max(n.ssi for n in nodes)
                max_csi = max(n.csi for n in nodes)
                if max_ssi > 0.3:
                    dom_process = "ab_h"
                elif max_csi > 0.4:
                    dom_process = "di_h"
                else:
                    dom_process = "as_h"

                all_lineages.append(FeatureLineage(
                    lineage_id=lineage_id_counter,
                    layer=layer,
                    depth=depth,
                    nodes=nodes,
                    founder_idx=founder_fid,
                    terminal_idx=nodes[-1].feature_idx if nodes[-1].checkpoint == checkpoint_labels[-1] else None,
                    dominant_process=dom_process,
                ))
                lineage_id_counter += 1

    # Sort by depth (longest first) and keep top 50 per layer
    lineages_by_layer = {}
    for lin in all_lineages:
        lineages_by_layer.setdefault(lin.layer, []).append(lin)

    top_lineages = []
    for layer in layers:
        layer_lins = lineages_by_layer.get(layer, [])
        layer_lins.sort(key=lambda x: x.depth, reverse=True)
        top_lineages.extend(layer_lins[:30])

    depths = [l.depth for l in all_lineages]
    surviving = sum(1 for l in all_lineages if l.terminal_idx is not None)

    stats = {
        "total_lineages": len(all_lineages),
        "mean_depth": round(float(np.mean(depths)), 2) if depths else 0,
        "max_depth": max(depths) if depths else 0,
        "surviving_to_terminal": surviving,
        "survival_rate": round(surviving / len(all_lineages), 4) if all_lineages else 0,
        "per_layer": {
            layer: {
                "total": len(lineages_by_layer.get(layer, [])),
                "surviving": sum(1 for l in lineages_by_layer.get(layer, []) if l.terminal_idx is not None),
            }
            for layer in layers
        }
    }

    return FeatureGenealogyData(
        lineages=top_lineages,
        layers=layers,
        stats=stats,
    )


def _compute_cross_layer_composition(sae_dir: Path, layers: list[str],
                                      checkpoint_labels: list[str],
                                      n_classes: int) -> CrossLayerCompositionData:
    """Compute cross-layer feature co-activation patterns."""
    import numpy as np

    # Adjacent layer pairs only
    layer_pairs = [(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
    pair_labels = [f"{a}-{b}" for a, b in layer_pairs]

    def compute_snapshot(ckpt: str) -> CrossLayerSnapshot:
        all_pairs = []
        density = {}

        for layer_a, layer_b in layer_pairs:
            act_a_path = sae_dir / "activations" / ckpt / f"{_safe_layer_filename(layer_a)}.npy"
            act_b_path = sae_dir / "activations" / ckpt / f"{_safe_layer_filename(layer_b)}.npy"

            if not act_a_path.exists() or not act_b_path.exists():
                continue

            try:
                acts_a = np.load(str(act_a_path))  # (n_samples, d_a)
                acts_b = np.load(str(act_b_path))  # (n_samples, d_b)
                n_samples = min(acts_a.shape[0], acts_b.shape[0])

                # Binary activity per sample
                active_a = (np.abs(acts_a[:n_samples]) > 1e-8)  # (n_samples, d_a)
                active_b = (np.abs(acts_b[:n_samples]) > 1e-8)  # (n_samples, d_b)

                # Top-N most active features per layer (by firing rate)
                top_n = 50
                rate_a = active_a.mean(axis=0)
                rate_b = active_b.mean(axis=0)
                top_a = np.argsort(rate_a)[-top_n:]
                top_b = np.argsort(rate_b)[-top_n:]

                sub_a = active_a[:, top_a]  # (n_samples, top_n)
                sub_b = active_b[:, top_b]  # (n_samples, top_n)

                # co_occ[i,j] = number of samples where both feature top_a[i] and top_b[j] are active
                co_occ = sub_a.astype(np.float64).T @ sub_b.astype(np.float64)  # (top_n, top_n)
                co_occ = np.nan_to_num(co_occ, nan=0.0, posinf=0.0, neginf=0.0)

                co_occ_rate = co_occ / max(n_samples, 1)

                # Rank pairs by excess co-occurrence (observed - rate_a * rate_b), which discounts
                # features that fire on most samples
                expected = np.outer(rate_a[top_a], rate_b[top_b])
                excess = co_occ_rate - expected

                flat_indices = np.argsort(excess.ravel())[-20:][::-1]

                pair_key = f"{layer_a}-{layer_b}"

                for flat_idx in flat_indices:
                    i, j = divmod(flat_idx, top_n)
                    exc_val = float(excess[i, j])
                    if exc_val < 0.01:
                        continue

                    all_pairs.append(CrossLayerPair(
                        layer_a=layer_a,
                        layer_b=layer_b,
                        feature_a=int(top_a[i]),
                        feature_b=int(top_b[j]),
                        co_occurrence_rate=round(float(co_occ_rate[i, j]), 4),
                        n_samples_both=int(co_occ[i, j]),
                        n_samples_total=n_samples,
                    ))

                # Mean excess co-occurrence for this layer pair
                density[pair_key] = round(float(excess.mean()), 6)

            except Exception as e:
                logger.warning(f"Error computing cross-layer for {ckpt}/{layer_a}-{layer_b}: {e}")
                continue

        # Sort by co-occurrence rate and keep top 30
        all_pairs.sort(key=lambda p: p.co_occurrence_rate, reverse=True)

        return CrossLayerSnapshot(
            checkpoint=ckpt,
            top_pairs=all_pairs[:30],
            layer_pair_density=density,
        )

    early = compute_snapshot(checkpoint_labels[0])
    terminal = compute_snapshot(checkpoint_labels[-1])

    return CrossLayerCompositionData(
        early=early,
        terminal=terminal,
        layer_pairs=pair_labels,
    )


def _compute_information_bottleneck(
    sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
    feature_landscape: dict, k_sparse: int
) -> InformationBottleneckData:
    """Track n_alive and effective dimensionality to test fit-then-compress."""
    points = []
    hidden_dim = 0

    for ckpt in checkpoint_labels:
        for layer in layers:
            fl = feature_landscape.get(ckpt, {}).get(layer, {})
            n_alive = fl.get('n_alive', 0)

            # Compute effective dimensionality from activation variances
            effective_dim = 0.0
            mean_act = 0.0
            act_file = sae_dir / "activations" / ckpt / f"{_safe_layer_filename(layer)}.npy"
            if act_file.exists():
                try:
                    H = np.load(str(act_file))
                    if hidden_dim == 0:
                        hidden_dim = H.shape[1]
                    # Per-feature variance across samples
                    variances = np.var(H, axis=0)
                    total_var = variances.sum()
                    if total_var > 1e-12:
                        # Participation ratio: (sum v_i)^2 / sum(v_i^2) - effective number of dimensions
                        effective_dim = float((total_var ** 2) / (variances ** 2).sum())
                    mean_act = float(np.abs(H).mean())
                except Exception:
                    pass

            compression = n_alive / max(hidden_dim, 1) if hidden_dim > 0 else 0.0
            points.append(InformationBottleneckPoint(
                checkpoint=ckpt, layer=layer,
                n_alive=n_alive, effective_dim=effective_dim,
                mean_activation=mean_act, compression_ratio=compression,
            ))

    return InformationBottleneckData(
        points=points, layers=layers,
        checkpoint_labels=checkpoint_labels, hidden_dim=hidden_dim,
    )


def _compute_superposition_evolution(
    feature_landscape: dict, co_occurrence: dict,
    layers: list[str], checkpoint_labels: list[str], k_sparse: int
) -> SuperpositionEvolutionData:
    """Track alive_ratio and excess co-occurrence to measure superposition."""
    points = []

    for ckpt in checkpoint_labels:
        for layer in layers:
            fl = feature_landscape.get(ckpt, {}).get(layer, {})
            n_alive = fl.get('n_alive', 0)
            alive_ratio = n_alive / max(k_sparse, 1)

            co = co_occurrence.get(ckpt, {}).get(layer, {})
            mean_excess = co.get('mean_excess_cooccurrence', 0.0)
            n_sig = co.get('n_significant_pairs', 0)

            points.append(SuperpositionEvolutionPoint(
                checkpoint=ckpt, layer=layer,
                alive_ratio=alive_ratio,
                mean_excess_cooccurrence=mean_excess,
                n_significant_pairs=n_sig,
            ))

    return SuperpositionEvolutionData(
        points=points, layers=layers,
        checkpoint_labels=checkpoint_labels, k_sparse=k_sparse,
    )


def _compute_forgettable_examples(
    sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
    selected_classes: list[str], n_classes: int,
    matching_method: str = "activation",
) -> ForgettableExamplesData:
    """Correlate per-sample churn (born+died) with terminal activation magnitude."""
    from scipy import stats as sp_stats

    transitions_dir = _transitions_dir(sae_dir, matching_method)
    transition_dirs = sorted([d for d in transitions_dir.iterdir() if d.is_dir()])

    n_samples_per_class = 0
    # Accumulate per-sample churn across all transitions and layers
    # sample_churn[global_sample_idx] = total born+died events
    sample_churn: dict[int, float] = {}

    for t_dir in transition_dirs:
        for layer in layers:
            layer_file = t_dir / f"{_safe_layer_filename(layer)}.json"
            if not layer_file.exists():
                continue
            try:
                with open(layer_file) as f:
                    entries = json.load(f)
                if n_samples_per_class == 0 and len(entries) > 0 and n_classes > 0:
                    n_samples_per_class = len(entries) // n_classes

                for entry in entries:
                    ci = entry.get('class_idx', 0)
                    si = entry.get('sample_idx_in_class', 0)
                    global_idx = ci * max(n_samples_per_class, 1) + si
                    n_born = len(entry.get('born', []))
                    n_died = len(entry.get('died', []))
                    sample_churn[global_idx] = sample_churn.get(global_idx, 0) + n_born + n_died
            except Exception:
                continue

    # Terminal activation magnitude per sample
    sample_terminal_act: dict[int, float] = {}
    terminal_ckpt = checkpoint_labels[-1] if checkpoint_labels else 'terminal'
    for layer in layers:
        act_file = sae_dir / "activations" / terminal_ckpt / f"{_safe_layer_filename(layer)}.npy"
        if not act_file.exists():
            continue
        try:
            H = np.load(str(act_file))
            for si in range(H.shape[0]):
                sample_terminal_act[si] = sample_terminal_act.get(si, 0) + float(np.abs(H[si]).mean())
        except Exception:
            continue

    # Per-class stats
    per_class = []
    for ci, cls_name in enumerate(selected_classes):
        start = ci * max(n_samples_per_class, 1)
        end = start + max(n_samples_per_class, 1)
        churns = [sample_churn.get(s, 0) for s in range(start, end)]
        acts = [sample_terminal_act.get(s, 0) for s in range(start, end)]
        per_class.append(ForgettableExampleEntry(
            class_idx=ci, class_name=cls_name,
            mean_churn=float(np.mean(churns)) if churns else 0,
            std_churn=float(np.std(churns)) if churns else 0,
            mean_terminal_activation=float(np.mean(acts)) if acts else 0,
            n_samples=len(churns),
        ))

    # Global correlation
    all_churns = []
    all_acts = []
    for si in sorted(set(list(sample_churn.keys()) + list(sample_terminal_act.keys()))):
        if si in sample_churn and si in sample_terminal_act:
            all_churns.append(sample_churn[si])
            all_acts.append(sample_terminal_act[si])

    corr, p_val = 0.0, 1.0
    if len(all_churns) > 10:
        try:
            corr, p_val = sp_stats.pearsonr(all_churns, all_acts)
            corr = float(corr)
            p_val = float(p_val)
        except Exception:
            pass

    # Top-20 most/least forgettable
    indexed = [(si, sample_churn.get(si, 0), sample_terminal_act.get(si, 0))
               for si in range(max(len(sample_churn), len(sample_terminal_act)))]
    indexed.sort(key=lambda x: x[1], reverse=True)

    high_churn = []
    for si, churn, act in indexed[:20]:
        ci = si // max(n_samples_per_class, 1) if n_samples_per_class > 0 else 0
        li = si % max(n_samples_per_class, 1) if n_samples_per_class > 0 else si
        cls_name = selected_classes[ci] if ci < len(selected_classes) else f"class_{ci}"
        high_churn.append({"sample_idx": si, "class_idx": ci, "class_name": cls_name,
                           "local_idx": li, "churn": churn, "terminal_activation": act})

    low_churn = []
    for si, churn, act in indexed[-20:]:
        ci = si // max(n_samples_per_class, 1) if n_samples_per_class > 0 else 0
        li = si % max(n_samples_per_class, 1) if n_samples_per_class > 0 else si
        cls_name = selected_classes[ci] if ci < len(selected_classes) else f"class_{ci}"
        low_churn.append({"sample_idx": si, "class_idx": ci, "class_name": cls_name,
                          "local_idx": li, "churn": churn, "terminal_activation": act})

    return ForgettableExamplesData(
        per_class=per_class,
        churn_activation_correlation=corr,
        churn_activation_p_value=p_val,
        high_churn_samples=high_churn,
        low_churn_samples=low_churn,
    )


def _compute_acquired_equivalence(
    sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
    selected_classes: list[str], class_to_sc: dict[str, str], n_classes: int
) -> AcquiredEquivalenceData:
    """Track within-superclass class-pair cosine similarity across checkpoints."""
    # Build superclass -> class indices mapping
    sc_to_classes: dict[str, list[int]] = {}
    for ci, cls_name in enumerate(selected_classes):
        sc = class_to_sc.get(cls_name, 'unknown')
        if sc not in sc_to_classes:
            sc_to_classes[sc] = []
        sc_to_classes[sc].append(ci)

    points = []
    between_sc_sims = []

    for ckpt in checkpoint_labels:
        # Build class centroid per layer (average activation across samples)
        class_centroids: dict[int, np.ndarray] = {}
        n_samples_per_class = 0

        for layer in layers:
            act_file = sae_dir / "activations" / ckpt / f"{_safe_layer_filename(layer)}.npy"
            if not act_file.exists():
                continue
            try:
                H = np.load(str(act_file))
                n_total = H.shape[0]
                n_per = n_total // max(n_classes, 1)
                if n_per == 0:
                    continue
                n_samples_per_class = n_per

                for ci in range(n_classes):
                    rows = H[ci * n_per: (ci + 1) * n_per]
                    centroid = rows.mean(axis=0)
                    if ci not in class_centroids:
                        class_centroids[ci] = centroid
                    else:
                        class_centroids[ci] = np.concatenate([class_centroids[ci], centroid])
            except Exception:
                continue

        if not class_centroids:
            continue

        # Compute within-SC pairwise cosine similarity
        def cosine(a, b):
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-12 or nb < 1e-12:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        for sc_name, class_indices in sc_to_classes.items():
            if len(class_indices) < 2:
                continue
            sims = []
            for i in range(len(class_indices)):
                for j in range(i + 1, len(class_indices)):
                    ci, cj = class_indices[i], class_indices[j]
                    if ci in class_centroids and cj in class_centroids:
                        sims.append(cosine(class_centroids[ci], class_centroids[cj]))
            if sims:
                points.append(AcquiredEquivalencePoint(
                    checkpoint=ckpt, superclass=sc_name,
                    mean_cosine=float(np.mean(sims)),
                ))

        # Between-SC similarity for contrast
        all_sc_names = list(sc_to_classes.keys())
        between_sims = []
        for i in range(len(all_sc_names)):
            for j in range(i + 1, len(all_sc_names)):
                for ci in sc_to_classes[all_sc_names[i]]:
                    for cj in sc_to_classes[all_sc_names[j]]:
                        if ci in class_centroids and cj in class_centroids:
                            between_sims.append(cosine(class_centroids[ci], class_centroids[cj]))
        if between_sims:
            between_sc_sims.append({
                "checkpoint": ckpt,
                "mean_cosine": float(np.mean(between_sims)),
            })

    return AcquiredEquivalenceData(
        points=points,
        superclass_names=list(sc_to_classes.keys()),
        checkpoint_labels=checkpoint_labels,
        between_sc_similarity=between_sc_sims,
    )


def _compute_layer_abstraction_progression(
    feature_landscape: dict, feature_matching: dict,
    layers: list[str], checkpoint_labels: list[str],
) -> LayerAbstractionProgressionData:
    """Examine how abstraction (SSI) and discrimination (CSI) develop across layers.

    Tests the literature prediction that deeper layers develop more abstract
    representations, and that developmental processes (Ab-H, Di-H, etc.)
    distribute differently across network depth.
    """
    from scipy import stats as sp_stats

    if not layers or not checkpoint_labels:
        return LayerAbstractionProgressionData()

    # Collect per-layer per-checkpoint landscape points
    points: list[LayerAbstractionPoint] = []
    for ckpt in checkpoint_labels:
        for layer in layers:
            fl = feature_landscape.get(ckpt, {}).get(layer, {})
            points.append(LayerAbstractionPoint(
                checkpoint=ckpt, layer=layer,
                mean_ssi=fl.get('mean_ssi', 0.0),
                mean_csi=fl.get('mean_csi', 0.0),
                n_alive=fl.get('n_alive', 0),
            ))

    # Per-layer process counts are approximated from feature_matching, which holds
    # born/died/stable/transformed counts per transition and layer
    # (process_intensity is aggregated across layers).
    layer_process_totals: dict[str, dict[str, int]] = {l: {"ab_h": 0, "tg_h": 0, "di_h": 0, "as_h": 0, "de_h": 0} for l in layers}

    for _pair_key, layer_dict in feature_matching.items():
        if not isinstance(layer_dict, dict):
            continue
        for layer_name, vals in layer_dict.items():
            if layer_name not in layer_process_totals or not isinstance(vals, dict):
                continue
            n_born = vals.get('n_born', 0)
            n_died = vals.get('n_died', 0)
            n_stable = vals.get('n_stable', 0)
            n_transformed = vals.get('n_transformed', 0)

            # Proxies: born -> Ab-H, transformed -> Di-H, died -> De-H, stable -> As-H
            layer_process_totals[layer_name]["ab_h"] += n_born
            layer_process_totals[layer_name]["di_h"] += n_transformed
            layer_process_totals[layer_name]["de_h"] += n_died
            layer_process_totals[layer_name]["as_h"] += n_stable

    # Layer totals are attached to the terminal checkpoint point
    terminal = checkpoint_labels[-1]
    for layer in layers:
        for pt in points:
            if pt.layer == layer and pt.checkpoint == terminal:
                totals = layer_process_totals[layer]
                pt.ab_h = totals["ab_h"]
                pt.tg_h = totals["tg_h"]
                pt.di_h = totals["di_h"]
                pt.as_h = totals["as_h"]
                pt.de_h = totals["de_h"]

    # Per-layer temporal gradients (Spearman of SSI/CSI vs checkpoint index)
    gradients: list[LayerAbstractionGradient] = []
    for layer in layers:
        layer_pts = [p for p in points if p.layer == layer]
        ssi_vals = [p.mean_ssi for p in layer_pts]
        csi_vals = [p.mean_csi for p in layer_pts]
        indices = list(range(len(layer_pts)))

        ssi_rho, ssi_p = 0.0, 1.0
        csi_rho, csi_p = 0.0, 1.0

        if len(indices) >= 3:
            try:
                r, p = sp_stats.spearmanr(indices, ssi_vals)
                ssi_rho, ssi_p = float(r), float(p)
            except Exception:
                pass
            try:
                r, p = sp_stats.spearmanr(indices, csi_vals)
                csi_rho, csi_p = float(r), float(p)
            except Exception:
                pass

        # Dominant process for this layer
        totals = layer_process_totals[layer]
        proc_sum = sum(totals.values())
        dominant = "none"
        dominant_pct = 0.0
        if proc_sum > 0:
            dominant = max(totals, key=totals.get)
            dominant_pct = totals[dominant] / proc_sum

        gradients.append(LayerAbstractionGradient(
            layer=layer,
            ssi_rho=ssi_rho, ssi_p=ssi_p,
            csi_rho=csi_rho, csi_p=csi_p,
            dominant_process=dominant, dominant_process_pct=dominant_pct,
        ))

    # Layer depth vs terminal selectivity (does deeper = more abstract?)
    terminal_ssi = []
    terminal_csi = []
    for layer in layers:
        fl = feature_landscape.get(terminal, {}).get(layer, {})
        terminal_ssi.append(fl.get('mean_ssi', 0.0))
        terminal_csi.append(fl.get('mean_csi', 0.0))

    layer_order_ssi_rho, layer_order_ssi_p = 0.0, 1.0
    layer_order_csi_rho, layer_order_csi_p = 0.0, 1.0
    depth_indices = list(range(len(layers)))

    if len(layers) >= 3:
        try:
            r, p = sp_stats.spearmanr(depth_indices, terminal_ssi)
            layer_order_ssi_rho, layer_order_ssi_p = float(r), float(p)
        except Exception:
            pass
        try:
            r, p = sp_stats.spearmanr(depth_indices, terminal_csi)
            layer_order_csi_rho, layer_order_csi_p = float(r), float(p)
        except Exception:
            pass

    return LayerAbstractionProgressionData(
        points=points,
        gradients=gradients,
        layers=layers,
        checkpoint_labels=checkpoint_labels,
        layer_order_ssi_rho=layer_order_ssi_rho,
        layer_order_ssi_p=layer_order_ssi_p,
        layer_order_csi_rho=layer_order_csi_rho,
        layer_order_csi_p=layer_order_csi_p,
    )


# Analysis 9b: Task-General Abstraction

def _compute_task_general_abstraction(
    feature_landscape: dict, feature_matching: dict, sae_data: dict,
    layers: list[str], checkpoint_labels: list[str],
) -> "TaskGeneralAbstractionData":
    """Analyse task-general (domain-wide) representations.

    Uses SAI (Super Abstraction Index) - features that fire uniformly across
    all superclasses and classes - to answer:
      1. Do DNNs develop shared computational primitives that transcend category structure?
      2. Which layers concentrate task-general vs category-specific abstraction?
      3. Do task-general features emerge before or after superclass-specific ones?
      4. Are task-general features more persistent than category-specific ones?
    """
    from scipy import stats as sp_stats

    if not layers or not checkpoint_labels:
        return TaskGeneralAbstractionData()

    # 1. SAI evolution points (per-layer per-checkpoint)
    sai_points: list = []
    for ckpt in checkpoint_labels:
        for layer in layers:
            fl = feature_landscape.get(ckpt, {}).get(layer, {})
            sai_points.append(TaskGeneralSaiPoint(
                checkpoint=ckpt, layer=layer,
                mean_sai=fl.get('mean_sai', 0.0),
                n_high_sai=fl.get('n_high_sai', 0),
                n_alive=fl.get('n_alive', 0),
            ))

    # 2. Process counts per layer from process_intensity and class_process_summary
    pi_list = sae_data.get('process_intensity', [])
    transitions: list[str] = []
    per_trans_tg_h: list[int] = []
    per_trans_ab_h: list[int] = []
    for pi in pi_list:
        if not isinstance(pi, dict):
            continue
        t = pi.get('transition', '')
        transitions.append(t)
        per_trans_tg_h.append(pi.get('tg_h', 0))
        per_trans_ab_h.append(pi.get('ab_h', 0))

    total_tg_h = sum(per_trans_tg_h)
    total_ab_h = sum(per_trans_ab_h)
    global_tg_share = total_tg_h / total_ab_h if total_ab_h > 0 else 0.0

    # 3. Per-layer profiles: SAI temporal trend + Tg-H vs Ab-H
    # Accumulate per-layer Tg-H and Ab-H counts from feature_matching process data
    layer_tg_h: dict[str, int] = {l: 0 for l in layers}
    layer_ab_h: dict[str, int] = {l: 0 for l in layers}

    class_process = sae_data.get('class_process_summary', {})
    for _ci_str, cls_data in class_process.items():
        per_layer = cls_data.get('per_layer', {})
        if not isinstance(per_layer, dict):
            continue
        for layer_name, layer_counts in per_layer.items():
            if layer_name not in layer_tg_h or not isinstance(layer_counts, dict):
                continue
            layer_tg_h[layer_name] += layer_counts.get('tg_h', 0)
            layer_ab_h[layer_name] += layer_counts.get('ab_h', 0)

    # If per_layer data not available, estimate from feature_matching
    if all(v == 0 for v in layer_tg_h.values()):
        for _pair_key, layer_dict in feature_matching.items():
            if not isinstance(layer_dict, dict):
                continue
            for layer_name, vals in layer_dict.items():
                if layer_name not in layer_tg_h or not isinstance(vals, dict):
                    continue
                n_born = vals.get('n_born', 0)
                # Born features split into Tg-H by the terminal mean SAI
                fl_term = feature_landscape.get(checkpoint_labels[-1], {}).get(layer_name, {})
                sai_ratio = fl_term.get('mean_sai', 0.0)
                est_tg = int(n_born * min(sai_ratio, 1.0))
                layer_tg_h[layer_name] += est_tg
                layer_ab_h[layer_name] += n_born

    layer_profiles: list = []
    for layer in layers:
        # SAI temporal trend (Spearman)
        layer_sai_vals = [
            feature_landscape.get(ckpt, {}).get(layer, {}).get('mean_sai', 0.0)
            for ckpt in checkpoint_labels
        ]
        sai_rho, sai_p = 0.0, 1.0
        if len(layer_sai_vals) >= 3:
            try:
                r, p = sp_stats.spearmanr(range(len(layer_sai_vals)), layer_sai_vals)
                sai_rho, sai_p = float(r), float(p)
            except Exception:
                pass

        tg_count = layer_tg_h[layer]
        ab_count = layer_ab_h[layer]
        tg_share = tg_count / ab_count if ab_count > 0 else 0.0

        # SAI x SSI correlation at terminal checkpoint
        fl_term = feature_landscape.get(checkpoint_labels[-1], {}).get(layer, {})
        feature_sai_list = fl_term.get('feature_sai', [])
        feature_ssi_list = fl_term.get('feature_ssi', [])
        sai_ssi_r, sai_ssi_p = 0.0, 1.0
        if len(feature_sai_list) >= 5 and len(feature_ssi_list) >= 5:
            import numpy as np
            sai_arr = np.array(feature_sai_list[:min(len(feature_sai_list), len(feature_ssi_list))])
            ssi_arr = np.array(feature_ssi_list[:len(sai_arr)])
            alive_mask = (sai_arr > 0) | (ssi_arr > 0)
            if alive_mask.sum() >= 5:
                try:
                    r, p = sp_stats.spearmanr(sai_arr[alive_mask], ssi_arr[alive_mask])
                    sai_ssi_r, sai_ssi_p = float(r), float(p)
                except Exception:
                    pass

        layer_profiles.append(TaskGeneralLayerProfile(
            layer=layer,
            terminal_sai=layer_sai_vals[-1] if layer_sai_vals else 0.0,
            sai_trend_rho=sai_rho, sai_trend_p=sai_p,
            tg_h_count=tg_count,
            ab_h_count=ab_count,
            tg_h_share=tg_share,
            sai_ssi_corr=sai_ssi_r,
            sai_ssi_p=sai_ssi_p,
        ))

    # 4. Update SAI points with Tg-H fraction
    for pt in sai_points:
        if not transitions:
            continue
        # Find the closest transition for this checkpoint
        ckpt_idx = checkpoint_labels.index(pt.checkpoint) if pt.checkpoint in checkpoint_labels else -1
        if ckpt_idx > 0 and ckpt_idx <= len(per_trans_tg_h):
            t_idx = ckpt_idx - 1  # transition i connects checkpoint i-1 -> i
            ab_val = per_trans_ab_h[t_idx] if t_idx < len(per_trans_ab_h) else 0
            tg_val = per_trans_tg_h[t_idx] if t_idx < len(per_trans_tg_h) else 0
            pt.tg_h_fraction = tg_val / ab_val if ab_val > 0 else 0.0

    # 5. Onset comparison: Tg-H vs superclass-specific Ab-H
    # ab_h_specific = ab_h - tg_h (only superclass-selective, not task-general)
    per_trans_ab_specific = [max(0, a - t) for a, t in zip(per_trans_ab_h, per_trans_tg_h)]

    tg_h_onset = -1
    ab_specific_onset = -1
    tg_cum: list[float] = []
    ab_spec_cum: list[float] = []
    tg_total = sum(per_trans_tg_h) or 1
    ab_spec_total = sum(per_trans_ab_specific) or 1
    tg_running, ab_running = 0, 0

    for t_idx in range(len(transitions)):
        tg_running += per_trans_tg_h[t_idx] if t_idx < len(per_trans_tg_h) else 0
        ab_running += per_trans_ab_specific[t_idx] if t_idx < len(per_trans_ab_specific) else 0
        tg_cum.append(tg_running / tg_total)
        ab_spec_cum.append(ab_running / ab_spec_total)
        if tg_h_onset < 0 and per_trans_tg_h[t_idx] > 0:
            tg_h_onset = t_idx
        if ab_specific_onset < 0 and per_trans_ab_specific[t_idx] > 0:
            ab_specific_onset = t_idx

    onset = TaskGeneralOnsetComparison(
        tg_h_onset_idx=tg_h_onset,
        ab_h_specific_onset_idx=ab_specific_onset,
        tg_h_cumulative=tg_cum,
        ab_h_specific_cumulative=ab_spec_cum,
        tg_h_leads=tg_h_onset >= 0 and (ab_specific_onset < 0 or tg_h_onset < ab_specific_onset),
    )

    # 6. Persistence analysis from feature genealogy
    # Use feature_matching born/died/stable to estimate survival rates by process type
    tg_survived, tg_total_born = 0, 0
    ab_spec_survived, ab_spec_total_born = 0, 0
    di_survived, di_total_born = 0, 0

    # Walk through feature_matching to count born features and their survival
    selectivity = sae_data.get('selectivity', {})
    for pair_key, layer_dict in feature_matching.items():
        if not isinstance(layer_dict, dict):
            continue
        for layer_name, vals in layer_dict.items():
            if not isinstance(vals, dict):
                continue
            n_born = vals.get('n_born', 0)
            n_stable = vals.get('n_stable', 0)
            n_died = vals.get('n_died', 0)
            # The layer's mean SAI estimates the task-general fraction of births
            fl_data = feature_landscape.get(checkpoint_labels[-1], {}).get(layer_name, {})
            mean_sai = fl_data.get('mean_sai', 0.0)
            mean_ssi = fl_data.get('mean_ssi', 0.0)
            mean_csi = fl_data.get('mean_csi', 0.0)

            est_tg_born = int(n_born * min(mean_sai * 2, 1.0))
            est_ab_spec_born = int(n_born * min(mean_ssi, 1.0)) - est_tg_born
            est_di_born = int(n_born * min(mean_csi, 1.0))
            survival_rate = n_stable / max(1, n_stable + n_died)

            tg_total_born += max(0, est_tg_born)
            tg_survived += int(max(0, est_tg_born) * survival_rate)
            ab_spec_total_born += max(0, est_ab_spec_born)
            ab_spec_survived += int(max(0, est_ab_spec_born) * survival_rate)
            di_total_born += max(0, est_di_born)
            di_survived += int(max(0, est_di_born) * survival_rate)

    persistence = TaskGeneralPersistence(
        tg_h_survival_rate=tg_survived / max(1, tg_total_born),
        ab_h_specific_survival_rate=ab_spec_survived / max(1, ab_spec_total_born),
        di_h_survival_rate=di_survived / max(1, di_total_born),
    )

    # 7. Depth correlation: does SAI increase with layer depth?
    terminal_sai_vals = [p.terminal_sai for p in layer_profiles]
    depth_sai_rho, depth_sai_p = 0.0, 1.0
    if len(terminal_sai_vals) >= 3:
        try:
            r, p = sp_stats.spearmanr(range(len(terminal_sai_vals)), terminal_sai_vals)
            depth_sai_rho, depth_sai_p = float(r), float(p)
        except Exception:
            pass

    # 8. Global SAI x SSI tradeoff
    all_sai, all_ssi = [], []
    for layer in layers:
        fl_term = feature_landscape.get(checkpoint_labels[-1], {}).get(layer, {})
        sai_list = fl_term.get('feature_sai', [])
        ssi_list = fl_term.get('feature_ssi', [])
        min_len = min(len(sai_list), len(ssi_list))
        all_sai.extend(sai_list[:min_len])
        all_ssi.extend(ssi_list[:min_len])

    global_sai_ssi_r, global_sai_ssi_p = 0.0, 1.0
    if len(all_sai) >= 10:
        import numpy as np
        sai_a = np.array(all_sai)
        ssi_a = np.array(all_ssi)
        alive = (sai_a > 0) | (ssi_a > 0)
        if alive.sum() >= 10:
            try:
                r, p = sp_stats.spearmanr(sai_a[alive], ssi_a[alive])
                global_sai_ssi_r, global_sai_ssi_p = float(r), float(p)
            except Exception:
                pass

    return TaskGeneralAbstractionData(
        sai_points=sai_points,
        layer_profiles=layer_profiles,
        onset_comparison=onset,
        persistence=persistence,
        layers=layers,
        checkpoint_labels=checkpoint_labels,
        transitions=transitions,
        total_tg_h=total_tg_h,
        total_ab_h=total_ab_h,
        global_tg_h_share=global_tg_share,
        sai_increases_with_depth=(depth_sai_rho > 0 and depth_sai_p < 0.05),
        depth_sai_rho=depth_sai_rho,
        depth_sai_p=depth_sai_p,
        global_sai_ssi_corr=global_sai_ssi_r,
        global_sai_ssi_p=global_sai_ssi_p,
    )


# Analysis 10: Critical Period Detection

def _compute_critical_period(
    sae_data: dict, checkpoint_labels: list[str],
    adj_per_class: Optional[dict] = None,
) -> CriticalPeriodData:
    """Detect whether developmental events concentrate in a critical window.

    Analogous to critical periods in biological neural development (Hubel & Wiesel).
    Computes event rate per transition, finds the window containing >50% of events,
    and tests whether it spans <40% of the training trajectory.
    """
    class_process = sae_data.get('class_process_summary', {})
    if not class_process:
        return CriticalPeriodData()

    # Transition list from the first class
    first_class = next(iter(class_process.values()), None)
    if not first_class or 'per_transition' not in first_class:
        return CriticalPeriodData()

    transitions = [t['transition'] for t in first_class['per_transition']]
    n_trans = len(transitions)
    if n_trans < 3:
        return CriticalPeriodData(transitions=transitions)

    # Aggregate events per transition across all classes
    per_trans = {proc: [0] * n_trans for proc in ['ab_h', 'tg_h', 'di_h', 'as_h', 'de_h']}
    per_trans_total = [0] * n_trans

    for ci_str, cls_data in class_process.items():
        pt_list = cls_data.get('per_transition', [])
        for t_idx in range(min(n_trans, len(pt_list))):
            for proc in ['ab_h', 'tg_h', 'di_h', 'as_h', 'de_h']:
                raw = pt_list[t_idx].get(proc, 0)
                # Subtract stable if requested
                if adj_per_class and proc in ('ab_h', 'tg_h', 'di_h'):
                    ci_int = int(ci_str) if isinstance(ci_str, str) else ci_str
                    t_adj = adj_per_class.get(ci_int, {}).get(transitions[t_idx], {})
                    raw = max(0, raw - t_adj.get(proc, 0))
                per_trans[proc][t_idx] += raw
                per_trans_total[t_idx] += raw

    grand_total = sum(per_trans_total)
    if grand_total == 0:
        return CriticalPeriodData(transitions=transitions, total_events=0)

    # Cumulative fraction and acceleration (second derivative)
    cumulative = []
    cum = 0
    rates = []
    for t_idx in range(n_trans):
        cum += per_trans_total[t_idx]
        cumulative.append(cum / grand_total)
        rates.append(per_trans_total[t_idx])

    # Acceleration = change in rate between consecutive transitions
    acceleration = [0.0] * n_trans
    for t_idx in range(1, n_trans - 1):
        acceleration[t_idx] = float(rates[t_idx + 1] - 2 * rates[t_idx] + rates[t_idx - 1])

    # Find critical window: smallest contiguous window containing >50% of events
    best_start, best_end = 0, n_trans - 1
    best_window_size = n_trans
    target_frac = 0.5

    for start in range(n_trans):
        window_sum = 0
        for end in range(start, n_trans):
            window_sum += per_trans_total[end]
            if window_sum / grand_total >= target_frac:
                window_size = end - start + 1
                if window_size < best_window_size:
                    best_window_size = window_size
                    best_start = start
                    best_end = end
                break

    critical_window_frac = best_window_size / n_trans
    events_in_window = sum(per_trans_total[best_start:best_end + 1])
    events_in_window_frac = events_in_window / grand_total

    # Peak transition
    peak_idx = rates.index(max(rates))

    # ab_h_specific = ab_h - tg_h per transition (needed by the per-process loop)
    per_trans['ab_h_specific'] = [max(0, a - t) for a, t in zip(per_trans['ab_h'], per_trans['tg_h'])]

    # Per-process critical windows
    per_process_window = {}
    for proc in ['ab_h', 'ab_h_specific', 'tg_h', 'di_h', 'as_h', 'de_h']:
        proc_total = sum(per_trans[proc])
        if proc_total == 0:
            continue
        p_best_start, p_best_end = 0, n_trans - 1
        p_best_size = n_trans
        for s in range(n_trans):
            w_sum = 0
            for e in range(s, n_trans):
                w_sum += per_trans[proc][e]
                if w_sum / proc_total >= target_frac:
                    if (e - s + 1) < p_best_size:
                        p_best_size = e - s + 1
                        p_best_start = s
                        p_best_end = e
                    break
        per_process_window[proc] = {
            'start_idx': p_best_start,
            'end_idx': p_best_end,
            'frac_in_window': sum(per_trans[proc][p_best_start:p_best_end + 1]) / proc_total,
            'window_frac': p_best_size / n_trans,
        }

    points = []
    for t_idx in range(n_trans):
        points.append(CriticalPeriodPoint(
            transition=transitions[t_idx],
            total_events=per_trans_total[t_idx],
            ab_h=per_trans['ab_h'][t_idx],
            tg_h=per_trans['tg_h'][t_idx],
            ab_h_specific=per_trans['ab_h_specific'][t_idx],
            di_h=per_trans['di_h'][t_idx],
            as_h=per_trans['as_h'][t_idx],
            de_h=per_trans['de_h'][t_idx],
            cumulative_frac=round(cumulative[t_idx], 4),
            acceleration=round(acceleration[t_idx], 2),
        ))

    is_critical = events_in_window_frac >= 0.5 and critical_window_frac < 0.4

    return CriticalPeriodData(
        points=points,
        transitions=transitions,
        critical_start_idx=best_start,
        critical_end_idx=best_end,
        critical_window_frac=round(critical_window_frac, 4),
        events_in_window_frac=round(events_in_window_frac, 4),
        peak_transition_idx=peak_idx,
        peak_transition_events=rates[peak_idx],
        total_events=grand_total,
        is_critical_period=is_critical,
        per_process_window=per_process_window,
    )


# Analysis 11: Representational Geometry Dynamics

def _compute_representational_geometry(
    sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
    n_classes: int,
) -> GeometryConvergenceData:
    """Track RDM evolution: Procrustes distance and CKA to terminal.

    Uses class-centroid RDMs (Kriegeskorte et al., 2008) to test whether
    representational geometry converges to a canonical structure.
    """
    import numpy as np

    if not layers or not checkpoint_labels or n_classes < 2:
        return GeometryConvergenceData()

    def _class_centroids(act_path: Path) -> Optional[np.ndarray]:
        """Load activations and compute per-class centroids."""
        if not act_path.exists():
            return None
        try:
            H = np.load(str(act_path))
            n_per = H.shape[0] // n_classes
            if n_per == 0:
                return None
            centroids = np.array([H[ci * n_per:(ci + 1) * n_per].mean(axis=0) for ci in range(n_classes)])
            return centroids
        except Exception:
            return None

    def _rdm(centroids: np.ndarray) -> np.ndarray:
        """Compute RDM (1 - cosine similarity) from centroids."""
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        normed = centroids / norms
        cos_sim = normed @ normed.T
        return 1.0 - cos_sim

    def _procrustes_distance(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
        """Procrustes distance between two RDMs (Frobenius norm of centered, scaled difference)."""
        a = rdm_a - rdm_a.mean()
        b = rdm_b - rdm_b.mean()
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 1.0
        a = a / norm_a
        b = b / norm_b
        return float(np.linalg.norm(a - b))

    def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
        """Compute linear CKA (Kornblith et al., 2019) between two representation matrices."""
        # X, Y: (n_samples, n_features) - here (n_classes, d_hidden)
        n = X.shape[0]
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)
        K = X.astype(np.float64) @ X.astype(np.float64).T  # (n, n)
        L = Y.astype(np.float64) @ Y.astype(np.float64).T  # (n, n)
        K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
        L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
        hsic_kl = np.trace(K @ L)
        hsic_kk = np.trace(K @ K)
        hsic_ll = np.trace(L @ L)
        denom = np.sqrt(max(hsic_kk * hsic_ll, 0.0))
        if denom < 1e-12 or not np.isfinite(denom):
            return 0.0
        result = float(hsic_kl / denom)
        return result if np.isfinite(result) else 0.0

    # Compute per-layer per-checkpoint centroids
    centroid_cache: dict[tuple[str, str], np.ndarray] = {}
    rdm_cache: dict[tuple[str, str], np.ndarray] = {}

    for ckpt in checkpoint_labels:
        for layer in layers:
            act_path = sae_dir / "activations" / ckpt / f"{_safe_layer_filename(layer)}.npy"
            c = _class_centroids(act_path)
            if c is not None:
                centroid_cache[(ckpt, layer)] = c
                rdm_cache[(ckpt, layer)] = _rdm(c)

    terminal = checkpoint_labels[-1]
    points = []

    for ckpt in checkpoint_labels:
        for layer in layers:
            rdm_curr = rdm_cache.get((ckpt, layer))
            rdm_term = rdm_cache.get((terminal, layer))
            cent_curr = centroid_cache.get((ckpt, layer))
            cent_term = centroid_cache.get((terminal, layer))

            procrustes_term = 0.0
            procrustes_prev = 0.0
            cka_term = 0.0

            if rdm_curr is not None and rdm_term is not None:
                procrustes_term = _procrustes_distance(rdm_curr, rdm_term)

            if cent_curr is not None and cent_term is not None:
                cka_term = _linear_cka(cent_curr, cent_term)

            # Procrustes to previous checkpoint
            ckpt_idx = checkpoint_labels.index(ckpt)
            if ckpt_idx > 0:
                prev_ckpt = checkpoint_labels[ckpt_idx - 1]
                rdm_prev = rdm_cache.get((prev_ckpt, layer))
                if rdm_curr is not None and rdm_prev is not None:
                    procrustes_prev = _procrustes_distance(rdm_curr, rdm_prev)

            points.append(RDMPoint(
                checkpoint=ckpt,
                layer=layer,
                procrustes_to_terminal=round(procrustes_term, 6),
                procrustes_to_previous=round(procrustes_prev, 6),
                cka_to_terminal=round(cka_term, 6),
            ))

    # Per-layer convergence: initial CKA to terminal
    per_layer_cka = {}
    for layer in layers:
        first_pt = next((p for p in points if p.layer == layer and p.checkpoint == checkpoint_labels[0]), None)
        per_layer_cka[layer] = first_pt.cka_to_terminal if first_pt else 0.0

    mean_final_cka = np.mean(list(per_layer_cka.values())) if per_layer_cka else 0.0

    # Check monotonic convergence: CKA to terminal should increase across checkpoints
    monotonic_count = 0
    total_checks = 0
    for layer in layers:
        layer_ckas = [p.cka_to_terminal for p in points if p.layer == layer]
        for i in range(1, len(layer_ckas)):
            total_checks += 1
            if layer_ckas[i] >= layer_ckas[i - 1] - 0.01:  # small tolerance
                monotonic_count += 1

    stabilizes = total_checks > 0 and (monotonic_count / total_checks) > 0.75

    return GeometryConvergenceData(
        points=points,
        layers=layers,
        checkpoint_labels=checkpoint_labels,
        per_layer_terminal_cka=per_layer_cka,
        mean_final_cka=round(float(mean_final_cka), 4),
        geometry_stabilizes=stabilizes,
    )


# Analysis 12: Process Interference / Facilitation

def _compute_process_interference(
    sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
    n_classes: int, matching_method: str = "activation",
) -> ProcessInterferenceData:
    """Test whether Ab-H and Di-H co-occur or anti-correlate at the sample level.

    Tracks both combined Ab-H (includes Tg-H) and ab_h_specific (Ab-H minus Tg-H)
    correlations with Di-H to separate superclass-specific vs task-general effects.
    """
    from scipy import stats as sp_stats

    transitions_dir = _transitions_dir(sae_dir, matching_method)
    if not transitions_dir.exists():
        return ProcessInterferenceData()

    transitions = []
    per_transition_points = []
    all_ab_h_counts = []
    all_ab_h_specific_counts = []
    all_tg_h_counts = []
    all_di_h_counts = []
    per_layer_ab = {l: [] for l in layers}
    per_layer_di = {l: [] for l in layers}

    for t_idx in range(len(checkpoint_labels) - 1):
        ckpt_a = checkpoint_labels[t_idx]
        ckpt_b = checkpoint_labels[t_idx + 1]
        pair_key = f"{ckpt_a}_to_{ckpt_b}"
        trans_label = f"{ckpt_a}->{ckpt_b}"
        transitions.append(trans_label)

        # Collect per-sample Ab-H, Tg-H and Di-H counts across all layers
        sample_processes: dict[str, dict[str, int]] = {}

        for layer in layers:
            layer_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not layer_file.exists():
                continue
            try:
                with open(layer_file) as f:
                    entries = json.load(f)
                for entry in entries:
                    ci = entry.get('class_idx', 0)
                    si = entry.get('sample_idx_in_class', 0)
                    key = f"{ci}_{si}"
                    if key not in sample_processes:
                        sample_processes[key] = {'ab_h': 0, 'tg_h': 0, 'di_h': 0}

                    for born_entry in entry.get('born', []):
                        procs = born_entry.get('processes', [])
                        if 'ab_h' in procs:
                            sample_processes[key]['ab_h'] += 1
                        if 'tg_h' in procs:
                            sample_processes[key]['tg_h'] += 1
                        if 'di_h' in procs:
                            sample_processes[key]['di_h'] += 1
            except Exception:
                continue

        # Per-layer correlation
        for layer in layers:
            layer_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not layer_file.exists():
                continue
            try:
                with open(layer_file) as f:
                    entries = json.load(f)
                layer_ab = []
                layer_di = []
                for entry in entries:
                    ab_count = 0
                    di_count = 0
                    for born_entry in entry.get('born', []):
                        procs = born_entry.get('processes', [])
                        if 'ab_h' in procs:
                            ab_count += 1
                        if 'di_h' in procs:
                            di_count += 1
                    layer_ab.append(ab_count)
                    layer_di.append(di_count)
                per_layer_ab[layer].extend(layer_ab)
                per_layer_di[layer].extend(layer_di)
            except Exception:
                continue

        # Compute per-transition correlation (combined and specific)
        if sample_processes:
            ab_vals = [v['ab_h'] for v in sample_processes.values()]
            tg_vals = [v['tg_h'] for v in sample_processes.values()]
            ab_specific_vals = [max(0, a - t) for a, t in zip(ab_vals, tg_vals)]
            di_vals = [v['di_h'] for v in sample_processes.values()]
            all_ab_h_counts.extend(ab_vals)
            all_ab_h_specific_counts.extend(ab_specific_vals)
            all_tg_h_counts.extend(tg_vals)
            all_di_h_counts.extend(di_vals)

            r, p = 0.0, 1.0
            r_spec, p_spec = 0.0, 1.0
            if len(ab_vals) >= 10 and len(set(ab_vals)) > 1 and len(set(di_vals)) > 1:
                try:
                    r, p = sp_stats.pearsonr(ab_vals, di_vals)
                    r, p = float(r), float(p)
                except Exception:
                    pass
            if len(ab_specific_vals) >= 10 and len(set(ab_specific_vals)) > 1 and len(set(di_vals)) > 1:
                try:
                    r_spec, p_spec = sp_stats.pearsonr(ab_specific_vals, di_vals)
                    r_spec, p_spec = float(r_spec), float(p_spec)
                except Exception:
                    pass

            per_transition_points.append(ProcessInterferencePoint(
                transition=trans_label,
                pearson_r=round(r, 4),
                p_value=round(p, 6),
                n_samples=len(sample_processes),
                mean_ab_h=round(float(np.mean(ab_vals)) if ab_vals else 0, 3),
                mean_di_h=round(float(np.mean(di_vals)) if di_vals else 0, 3),
                pearson_r_specific=round(r_spec, 4),
                p_value_specific=round(p_spec, 6),
                mean_ab_h_specific=round(float(np.mean(ab_specific_vals)) if ab_specific_vals else 0, 3),
                mean_tg_h=round(float(np.mean(tg_vals)) if tg_vals else 0, 3),
            ))
        else:
            per_transition_points.append(ProcessInterferencePoint(transition=trans_label))

    # Overall correlation (combined)
    overall_r, overall_p = 0.0, 1.0
    if len(all_ab_h_counts) >= 20:
        try:
            overall_r, overall_p = sp_stats.pearsonr(all_ab_h_counts, all_di_h_counts)
            overall_r, overall_p = float(overall_r), float(overall_p)
        except Exception:
            pass

    # Overall correlation (ab_h_specific vs di_h)
    overall_r_spec, overall_p_spec = 0.0, 1.0
    if len(all_ab_h_specific_counts) >= 20:
        try:
            overall_r_spec, overall_p_spec = sp_stats.pearsonr(all_ab_h_specific_counts, all_di_h_counts)
            overall_r_spec, overall_p_spec = float(overall_r_spec), float(overall_p_spec)
        except Exception:
            pass

    # Per-layer correlation
    per_layer_results = {}
    for layer in layers:
        if len(per_layer_ab[layer]) >= 20:
            try:
                r, p = sp_stats.pearsonr(per_layer_ab[layer], per_layer_di[layer])
                per_layer_results[layer] = {'r': round(float(r), 4), 'p': round(float(p), 6)}
            except Exception:
                per_layer_results[layer] = {'r': 0.0, 'p': 1.0}

    is_interference = overall_r < -0.1 and overall_p < 0.05
    is_facilitation = overall_r > 0.1 and overall_p < 0.05
    is_interference_spec = overall_r_spec < -0.1 and overall_p_spec < 0.05
    is_facilitation_spec = overall_r_spec > 0.1 and overall_p_spec < 0.05

    return ProcessInterferenceData(
        per_transition=per_transition_points,
        transitions=transitions,
        overall_r=round(overall_r, 4),
        overall_p=round(overall_p, 6),
        is_interference=is_interference,
        is_facilitation=is_facilitation,
        per_layer=per_layer_results,
        overall_r_specific=round(overall_r_spec, 4),
        overall_p_specific=round(overall_p_spec, 6),
        is_interference_specific=is_interference_spec,
        is_facilitation_specific=is_facilitation_spec,
    )


# Analysis 13: Catastrophic Forgetting Signatures

def _compute_catastrophic_forgetting(
    sae_data: dict, selected_classes: list[str],
    adj_per_class: Optional[dict] = None,
) -> CatastrophicForgettingData:
    """Detect whether learning in one class causes feature death in others.

    Computes per-class per-transition De-H events and correlates with
    Di-H events (new learning) in other classes at the same transition.
    """
    from scipy import stats as sp_stats

    class_process = sae_data.get('class_process_summary', {})
    if not class_process:
        return CatastrophicForgettingData()

    first_class = next(iter(class_process.values()), None)
    if not first_class or 'per_transition' not in first_class:
        return CatastrophicForgettingData()

    transitions = [t['transition'] for t in first_class['per_transition']]
    n_trans = len(transitions)
    n_classes = len(selected_classes)

    if n_trans == 0 or n_classes < 2:
        return CatastrophicForgettingData(transitions=transitions, class_names=selected_classes)

    # Build per-class per-transition process counts
    # class_events[ci][t_idx] = {ab_h, di_h, as_h, de_h}
    class_events = {}
    for ci in range(n_classes):
        cls_data = class_process.get(str(ci), {})
        pt_list = cls_data.get('per_transition', [])
        class_events[ci] = []
        for t_idx in range(n_trans):
            if t_idx < len(pt_list):
                counts = {}
                for proc in ['ab_h', 'tg_h', 'di_h', 'as_h', 'de_h']:
                    raw = pt_list[t_idx].get(proc, 0)
                    if adj_per_class and proc in ('ab_h', 'tg_h', 'di_h'):
                        t_adj = adj_per_class.get(ci, {}).get(transitions[t_idx], {})
                        raw = max(0, raw - t_adj.get(proc, 0))
                    counts[proc] = raw
                class_events[ci].append(counts)
            else:
                class_events[ci].append({'ab_h': 0, 'tg_h': 0, 'di_h': 0, 'as_h': 0, 'de_h': 0})

    # For each class at each transition: own De-H vs others' Di-H + born
    points = []
    own_de_h_all = []
    others_di_h_all = []

    for ci in range(n_classes):
        for t_idx in range(n_trans):
            own_de = class_events[ci][t_idx]['de_h']
            others_di = sum(class_events[cj][t_idx]['di_h'] for cj in range(n_classes) if cj != ci)
            others_born = sum(
                class_events[cj][t_idx]['ab_h'] + class_events[cj][t_idx]['as_h']
                for cj in range(n_classes) if cj != ci
            )

            points.append(ForgettingTransitionPoint(
                transition=transitions[t_idx],
                class_idx=ci,
                class_name=selected_classes[ci] if ci < len(selected_classes) else f"class_{ci}",
                own_de_h=own_de,
                others_di_h=others_di,
                others_born=others_born,
            ))

            own_de_h_all.append(own_de)
            others_di_h_all.append(others_di)

    # Global correlation
    corr, p_val = 0.0, 1.0
    if len(own_de_h_all) >= 10 and max(own_de_h_all) > 0:
        try:
            corr, p_val = sp_stats.pearsonr(own_de_h_all, others_di_h_all)
            corr, p_val = float(corr), float(p_val)
        except Exception:
            pass

    # Per-class vulnerability: mean De-H per transition
    per_class_vuln = []
    for ci in range(n_classes):
        de_h_vals = [class_events[ci][t]['de_h'] for t in range(n_trans)]
        mean_de = float(np.mean(de_h_vals)) if de_h_vals else 0
        # Vulnerability: correlation between this class's De-H and total others' Di-H
        own = [class_events[ci][t]['de_h'] for t in range(n_trans)]
        others = [
            sum(class_events[cj][t]['di_h'] for cj in range(n_classes) if cj != ci)
            for t in range(n_trans)
        ]
        vuln_r = 0.0
        if n_trans >= 3 and max(own) > 0:
            try:
                vuln_r, _ = sp_stats.pearsonr(own, others)
                vuln_r = float(vuln_r)
            except Exception:
                pass

        per_class_vuln.append({
            'class_idx': ci,
            'class_name': selected_classes[ci] if ci < len(selected_classes) else f"class_{ci}",
            'mean_de_h': round(mean_de, 2),
            'vulnerability_score': round(vuln_r, 4),
        })

    # Most vulnerable (highest mean De-H) and most disruptive (highest Di-H when others die)
    most_vulnerable = max(per_class_vuln, key=lambda x: x['mean_de_h'])['class_name'] if per_class_vuln else ""

    # Most disruptive: class with highest total Di-H events
    di_h_totals = {}
    for ci in range(n_classes):
        di_h_totals[ci] = sum(class_events[ci][t]['di_h'] for t in range(n_trans))
    if di_h_totals:
        most_disruptive_ci = max(di_h_totals, key=di_h_totals.get)
        most_disruptive = selected_classes[most_disruptive_ci] if most_disruptive_ci < len(selected_classes) else ""
    else:
        most_disruptive = ""

    is_catastrophic = corr > 0.15 and p_val < 0.05

    return CatastrophicForgettingData(
        points=points,
        transitions=transitions,
        class_names=selected_classes,
        cross_class_correlation=round(corr, 4),
        cross_class_p_value=round(p_val, 6),
        is_catastrophic=is_catastrophic,
        per_class_vulnerability=per_class_vuln,
        most_vulnerable_class=most_vulnerable,
        most_disruptive_class=most_disruptive,
    )


def _compute_feature_survival(
    sae_dir: Path, layers: list[str], checkpoint_labels: list[str],
    sae_data: dict, matching_method: str = "activation",
) -> Optional[FeatureSurvivalData]:
    """Kaplan-Meier survival analysis of SAE features across training.

    For each feature alive at each checkpoint, track how many subsequent
    transitions it survives (remains in the 'stable' set). Features that
    reach the terminal checkpoint are right-censored. Stratify by initial
    selectivity (high-SSI vs low-SSI) and test whether selectivity predicts
    longer survival (Spearman correlation + log-rank test).
    """
    import numpy as np
    from scipy import stats

    if len(checkpoint_labels) < 3:
        return None

    transitions_dir = _transitions_dir(sae_dir, matching_method)
    selectivity = sae_data.get('selectivity', {})
    n_transitions = len(checkpoint_labels) - 1

    # Step 1: Build per-layer forward maps (same as genealogy)
    # For each layer, build: transition_idx -> {src_fid: dst_fid}
    # and track which features are alive at each checkpoint.

    # Collect per-feature records: (layer, birth_ckpt_idx, lifespan, initial_ssi, initial_csi)
    feature_records: list[dict] = []

    for layer in layers:
        # Build forward maps: ckpt_idx -> {src_fid: dst_fid}
        forward_maps: list[dict[int, int]] = [dict() for _ in range(n_transitions)]

        for t_idx in range(n_transitions):
            ckpt_a = checkpoint_labels[t_idx]
            ckpt_b = checkpoint_labels[t_idx + 1]
            pair_key = f"{ckpt_a}_to_{ckpt_b}"
            trans_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not trans_file.exists():
                continue
            try:
                with open(trans_file, 'r') as f:
                    samples = json.load(f)
            except Exception:
                continue

            # Count stable links across samples, pick majority link per source
            link_counts: dict[tuple[int, int], int] = {}
            for sample in samples:
                for entry in sample.get('stable', []):
                    fid_a = entry.get('fid_a')
                    fid_b = entry.get('fid_b')
                    if fid_a is not None and fid_b is not None:
                        key = (fid_a, fid_b)
                        link_counts[key] = link_counts.get(key, 0) + 1

            src_to_best: dict[int, int] = {}
            for (fid_a, fid_b), count in link_counts.items():
                if fid_a not in src_to_best:
                    src_to_best[fid_a] = fid_b
                else:
                    # Check if this link is stronger
                    old_b = src_to_best[fid_a]
                    old_count = link_counts.get((fid_a, old_b), 0)
                    if count > old_count:
                        src_to_best[fid_a] = fid_b

            forward_maps[t_idx] = src_to_best

        # Step 2: For each checkpoint, find alive features and trace forward
        for birth_idx in range(len(checkpoint_labels)):
            ckpt_label = checkpoint_labels[birth_idx]

            # Get alive features at this checkpoint
            sel_ckpt = selectivity.get(ckpt_label, {}).get(layer, {})
            ssi_list = sel_ckpt.get('feature_ssi', [])
            csi_list = sel_ckpt.get('feature_csi', [])
            if not ssi_list:
                continue

            # Find alive features (those with non-trivial SSI or that appear in forward map)
            alive_fids: set[int] = set()
            if birth_idx < n_transitions:
                alive_fids = set(forward_maps[birth_idx].keys())
            # Also add features from selectivity that have any activity
            for fid in range(len(ssi_list)):
                if ssi_list[fid] > 0 or (fid < len(csi_list) and csi_list[fid] > 0):
                    alive_fids.add(fid)

            # Trace each alive feature forward
            for fid in alive_fids:
                current_fid = fid
                lifespan = 0
                for t_idx in range(birth_idx, n_transitions):
                    fwd = forward_maps[t_idx]
                    if current_fid in fwd:
                        lifespan += 1
                        current_fid = fwd[current_fid]
                    else:
                        break  # feature died

                survived_to_terminal = (birth_idx + lifespan) >= n_transitions
                initial_ssi = ssi_list[fid] if fid < len(ssi_list) else 0.0
                initial_csi = csi_list[fid] if fid < len(csi_list) else 0.0
                max_possible = n_transitions - birth_idx

                feature_records.append({
                    'layer': layer,
                    'birth_idx': birth_idx,
                    'lifespan': lifespan,
                    'max_possible': max_possible,
                    'censored': survived_to_terminal,
                    'initial_ssi': initial_ssi,
                    'initial_csi': initial_csi,
                })

    if not feature_records:
        return None

    # Step 3: Build overall Kaplan-Meier curve
    # Pool all features, normalize lifespan relative to max possible
    transition_labels = [f"{checkpoint_labels[i]}->{checkpoint_labels[i+1]}" for i in range(n_transitions)]

    def _build_km_curve(records: list[dict]) -> list[SurvivalCurvePoint]:
        """Build KM curve from feature records.

        Time axis = number of transitions survived (0..n_transitions).
        """
        if not records:
            return []

        max_time = max(r['max_possible'] for r in records)
        n_at_risk = len(records)
        survival = 1.0
        curve: list[SurvivalCurvePoint] = []

        for t in range(max_time + 1):
            # Features that died at exactly time t (lifespan == t and not censored)
            n_events = sum(1 for r in records if r['lifespan'] == t and not r['censored'])
            # Features censored at time t (survived to terminal with lifespan == t)
            n_censored = sum(1 for r in records if r['lifespan'] == t and r['censored'])

            if n_at_risk > 0 and n_events > 0:
                survival *= (1.0 - n_events / n_at_risk)

            trans_label = transition_labels[t] if t < len(transition_labels) else f"t={t}"
            curve.append(SurvivalCurvePoint(
                transition_idx=t,
                transition=trans_label,
                n_at_risk=n_at_risk,
                n_events=n_events,
                n_censored=n_censored,
                survival_prob=round(survival, 4),
            ))

            n_at_risk -= (n_events + n_censored)
            if n_at_risk <= 0:
                break

        return curve

    def _median_survival(curve: list[SurvivalCurvePoint]) -> Optional[float]:
        for pt in curve:
            if pt.survival_prob <= 0.5:
                return float(pt.transition_idx)
        return None

    overall_curve = _build_km_curve(feature_records)
    overall_median = _median_survival(overall_curve)

    # Step 4: Stratify by SSI
    ssi_values = [r['initial_ssi'] for r in feature_records]
    ssi_median = float(np.median(ssi_values)) if ssi_values else 0.3

    high_ssi = [r for r in feature_records if r['initial_ssi'] >= ssi_median]
    low_ssi = [r for r in feature_records if r['initial_ssi'] < ssi_median]

    high_ssi_curve = _build_km_curve(high_ssi)
    low_ssi_curve = _build_km_curve(low_ssi)

    # Also stratify by CSI
    csi_values = [r['initial_csi'] for r in feature_records]
    csi_median = float(np.median(csi_values)) if csi_values else 0.4

    high_csi = [r for r in feature_records if r['initial_csi'] >= csi_median]
    low_csi = [r for r in feature_records if r['initial_csi'] < csi_median]

    high_csi_curve = _build_km_curve(high_csi)
    low_csi_curve = _build_km_curve(low_csi)

    # Low-selectivity cohort (both below median)
    low_both = [r for r in feature_records if r['initial_ssi'] < ssi_median and r['initial_csi'] < csi_median]
    low_both_curve = _build_km_curve(low_both)

    cohorts = [
        SurvivalCohort(cohort_name="High SSI", n_features=len(high_ssi),
                        curve=high_ssi_curve, median_survival=_median_survival(high_ssi_curve)),
        SurvivalCohort(cohort_name="Low SSI", n_features=len(low_ssi),
                        curve=low_ssi_curve, median_survival=_median_survival(low_ssi_curve)),
        SurvivalCohort(cohort_name="High CSI", n_features=len(high_csi),
                        curve=high_csi_curve, median_survival=_median_survival(high_csi_curve)),
        SurvivalCohort(cohort_name="Low CSI", n_features=len(low_csi),
                        curve=low_csi_curve, median_survival=_median_survival(low_csi_curve)),
        SurvivalCohort(cohort_name="Low SSI+CSI", n_features=len(low_both),
                        curve=low_both_curve, median_survival=_median_survival(low_both_curve)),
    ]

    # Step 5: Log-rank test (high-SSI vs low-SSI)
    logrank_chi2 = None
    logrank_p = None
    if high_ssi and low_ssi:
        try:
            # Manual log-rank: compare observed vs expected deaths at each time
            max_time = max(r['max_possible'] for r in feature_records)
            chi2_sum = 0.0
            var_sum = 0.0

            for t in range(max_time + 1):
                n1 = sum(1 for r in high_ssi if r['lifespan'] >= t)
                n2 = sum(1 for r in low_ssi if r['lifespan'] >= t)
                d1 = sum(1 for r in high_ssi if r['lifespan'] == t and not r['censored'])
                d2 = sum(1 for r in low_ssi if r['lifespan'] == t and not r['censored'])
                n = n1 + n2
                d = d1 + d2

                if n > 1 and d > 0:
                    e1 = d * n1 / n  # expected deaths in group 1
                    var = d * (n - d) * n1 * n2 / (n * n * max(1, n - 1))
                    chi2_sum += (d1 - e1)
                    var_sum += var

            if var_sum > 0:
                logrank_chi2 = round(chi2_sum ** 2 / var_sum, 4)
                logrank_p = round(float(1.0 - stats.chi2.cdf(logrank_chi2, df=1)), 6)
        except Exception:
            pass

    # Step 6: Spearman correlations (selectivity vs lifespan)
    lifespans = np.array([r['lifespan'] for r in feature_records])
    initial_ssis = np.array([r['initial_ssi'] for r in feature_records])
    initial_csis = np.array([r['initial_csi'] for r in feature_records])

    ssi_corr, ssi_p = None, None
    csi_corr, csi_p = None, None
    if len(lifespans) > 10:
        try:
            rho, p = stats.spearmanr(initial_ssis, lifespans)
            ssi_corr = round(float(rho), 4)
            ssi_p = round(float(p), 6)
        except Exception:
            pass
        try:
            rho, p = stats.spearmanr(initial_csis, lifespans)
            csi_corr = round(float(rho), 4)
            csi_p = round(float(p), 6)
        except Exception:
            pass

    # Step 7: Per-layer summary
    per_layer: list[LayerSurvivalSummary] = []
    for layer in layers:
        layer_recs = [r for r in feature_records if r['layer'] == layer]
        if not layer_recs:
            per_layer.append(LayerSurvivalSummary(layer=layer))
            continue
        ls = [r['lifespan'] for r in layer_recs]
        n_surviving = sum(1 for r in layer_recs if r['censored'])
        per_layer.append(LayerSurvivalSummary(
            layer=layer,
            n_tracked=len(layer_recs),
            survival_rate=round(n_surviving / len(layer_recs), 4),
            median_lifespan=round(float(np.median(ls)), 2),
            mean_lifespan=round(float(np.mean(ls)), 2),
        ))

    n_surviving_total = sum(1 for r in feature_records if r['censored'])

    return FeatureSurvivalData(
        overall_curve=overall_curve,
        cohorts=cohorts,
        per_layer=per_layer,
        layers=layers,
        checkpoint_labels=checkpoint_labels,
        transitions=transition_labels,
        n_total_tracked=len(feature_records),
        overall_survival_rate=round(n_surviving_total / len(feature_records), 4) if feature_records else 0.0,
        overall_median_lifespan=overall_median if overall_median is not None else 0.0,
        logrank_chi2=logrank_chi2,
        logrank_p=logrank_p,
        ssi_survival_corr=ssi_corr,
        ssi_survival_p=ssi_p,
        csi_survival_corr=csi_corr,
        csi_survival_p=csi_p,
    )


def _compute_polysemanticity_dynamics(
    sae_dir: Path, sae_data: dict, layers: list[str],
    checkpoint_labels: list[str], matching_method: str,
    pi_threshold: float = 0.2,
) -> PolysemantricityDynamicsData:
    """Track polysemantic-to-monosemantic feature transitions during training.

    Uses SAI (Super Abstraction Index = normalized class entropy) as the
    Polysemanticity Index (PI). PI=1 means the feature fires uniformly across
    all classes (maximally polysemantic); PI=0 means it fires for only one
    class (monosemantic).
    """
    import numpy as np

    selectivity = sae_data.get('selectivity', {})
    if not selectivity:
        return PolysemantricityDynamicsData()

    # Phase 1: PI distribution per checkpoint-layer
    pi_distribution = []
    # Cache feature_sai arrays for sharpening phase
    sai_cache: dict[tuple[str, str], list[float]] = {}

    for ckpt in checkpoint_labels:
        for layer in layers:
            sel = selectivity.get(ckpt, {}).get(layer, {})
            feature_sai = sel.get('feature_sai', [])
            if not feature_sai:
                continue
            arr = np.array(feature_sai, dtype=float)
            sai_cache[(ckpt, layer)] = feature_sai
            # Active features: SAI > 0 implies the feature fires on at least one sample
            active = arr[arr > 0]
            if len(active) == 0:
                continue
            pi_distribution.append(PolysemanticityCkptPoint(
                checkpoint=ckpt, layer=layer,
                mean_pi=float(np.mean(active)),
                median_pi=float(np.median(active)),
                q25_pi=float(np.percentile(active, 25)),
                q75_pi=float(np.percentile(active, 75)),
                n_active=int(len(active)),
            ))

    # Phase 2: Sharpening / blurring events via stable features
    transitions_dir = _transitions_dir(sae_dir, matching_method)
    sharpening_events = []
    born_pi_profiles = []
    all_transitions = []
    total_sharp = 0
    total_blur = 0

    for t_idx in range(len(checkpoint_labels) - 1):
        ckpt_a = checkpoint_labels[t_idx]
        ckpt_b = checkpoint_labels[t_idx + 1]
        pair_key = f"{ckpt_a}_to_{ckpt_b}"
        trans_label = f"{ckpt_a}->{ckpt_b}"
        all_transitions.append(trans_label)

        for layer in layers:
            layer_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not layer_file.exists():
                continue

            sai_a = sai_cache.get((ckpt_a, layer), [])
            sai_b = sai_cache.get((ckpt_b, layer), [])
            if not sai_a or not sai_b:
                continue

            try:
                with open(layer_file) as f:
                    entries = json.load(f)
            except Exception:
                continue

            n_sharp = 0
            n_blur = 0
            deltas = []
            born_pis = []
            pop_mean_pi = float(np.mean([v for v in sai_b if v > 0])) if any(v > 0 for v in sai_b) else 0.0

            for entry in entries:
                # Stable features: track PI change
                for stable in entry.get('stable', []):
                    fid_a = stable.get('fid_a', 0)
                    fid_b = stable.get('fid_b', 0)
                    if fid_a < len(sai_a) and fid_b < len(sai_b):
                        pi_a = sai_a[fid_a]
                        pi_b = sai_b[fid_b]
                        delta = pi_b - pi_a
                        deltas.append(delta)
                        if delta < -pi_threshold:
                            n_sharp += 1
                        elif delta > pi_threshold:
                            n_blur += 1

                # Born features: record their PI
                for born in entry.get('born', []):
                    fid = born.get('fid', 0)
                    if fid < len(sai_b) and sai_b[fid] > 0:
                        born_pis.append(sai_b[fid])

            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            sharpening_events.append(SharpeningTransitionPoint(
                transition=trans_label, layer=layer,
                n_sharpening=n_sharp, n_blurring=n_blur,
                net_sharpening=n_sharp - n_blur,
                mean_pi_delta=mean_delta,
            ))
            total_sharp += n_sharp
            total_blur += n_blur

            # Born PI profile
            if born_pis:
                born_mean = float(np.mean(born_pis))
                born_pi_profiles.append(BornPIProfile(
                    transition=trans_label, layer=layer,
                    born_mean_pi=born_mean,
                    population_mean_pi=pop_mean_pi,
                    born_more_mono=born_mean < pop_mean_pi,
                ))

    # Phase 3: Layer depth PI at terminal
    layer_depth_pi = []
    terminal_ckpt = checkpoint_labels[-1] if checkpoint_labels else ""
    for i, layer in enumerate(layers):
        sel = selectivity.get(terminal_ckpt, {}).get(layer, {})
        feature_sai = sel.get('feature_sai', [])
        if feature_sai:
            arr = np.array(feature_sai, dtype=float)
            active = arr[arr > 0]
            mean_pi = float(np.mean(active)) if len(active) > 0 else 0.0
        else:
            mean_pi = 0.0
        layer_depth_pi.append(LayerDepthPIPoint(
            layer=layer, layer_idx=i, mean_pi=mean_pi,
        ))

    # Outcome label
    net = total_sharp - total_blur
    if net > 0:
        verdict = "Net Sharpening"
    elif net < 0:
        verdict = "Net Blurring"
    else:
        verdict = "Stable Polysemanticity"

    return PolysemantricityDynamicsData(
        pi_distribution=pi_distribution,
        sharpening_events=sharpening_events,
        born_pi_profiles=born_pi_profiles,
        layer_depth_pi=layer_depth_pi,
        layers=layers,
        checkpoints=checkpoint_labels,
        transitions=all_transitions,
        total_sharpening=total_sharp,
        total_blurring=total_blur,
        net_sharpening=net,
        verdict=verdict,
        pi_threshold=pi_threshold,
    )


def _compute_representational_drift(
    sae_dir: Path, sae_data: dict, layers: list[str],
    checkpoint_labels: list[str], matching_method: str,
    pi_list: list,
    accuracy_plateau_threshold: float = 0.01,
) -> RepresentationalDriftData:
    """Analyze feature turnover after validation accuracy converges.

    Tests whether internal representations keep changing ("silent drift")
    after external performance plateaus (Rule et al., 2019; Driscoll et al., 2022).
    """
    import numpy as np

    if len(checkpoint_labels) < 3:
        return RepresentationalDriftData(drift_verdict="Insufficient Data")

    # Phase 1: Convergence detection
    # Convergence from val_accuracy in the enriched pi_list
    accuracies = []
    for pi in pi_list:
        acc = getattr(pi, 'end_val_accuracy', None)
        if acc is None and isinstance(pi, dict):
            acc = pi.get('end_val_accuracy')
        accuracies.append(acc)

    convergence_idx = -1
    convergence_method = "no_accuracy_data"
    convergence_acc = None

    if any(a is not None for a in accuracies):
        convergence_method = "accuracy_plateau_1pct"
        # Find first transition after which all subsequent improvements < threshold
        for i in range(len(accuracies)):
            if accuracies[i] is None:
                continue
            all_plateau = True
            for j in range(i + 1, len(accuracies)):
                if accuracies[j] is not None and accuracies[i] is not None:
                    if (accuracies[j] - accuracies[i]) > accuracy_plateau_threshold:
                        all_plateau = False
                        break
            if all_plateau and i < len(accuracies) - 1:
                convergence_idx = i
                convergence_acc = accuracies[i]
                break

    n_trans = len(checkpoint_labels) - 1
    transitions_dir = _transitions_dir(sae_dir, matching_method)
    feature_matching = sae_data.get('feature_matching', {})
    feature_landscape = sae_data.get('feature_landscape', {})

    # Phase 2: Per-transition churn with selectivity
    drift_points = []
    all_transitions = []
    # Buckets for pre/post selectivity aggregation
    pre_born_ssi, pre_born_csi = [], []
    post_born_ssi, post_born_csi = [], []
    pre_died_ssi, pre_died_csi = [], []
    post_died_ssi, post_died_csi = [], []
    # Per-layer churn tracking
    layer_pre_churn: dict[str, list[int]] = {l: [] for l in layers}
    layer_post_churn: dict[str, list[int]] = {l: [] for l in layers}

    for t_idx in range(n_trans):
        ckpt_a = checkpoint_labels[t_idx]
        ckpt_b = checkpoint_labels[t_idx + 1]
        pair_key = f"{ckpt_a}_to_{ckpt_b}"
        trans_label = f"{ckpt_a}->{ckpt_b}"
        all_transitions.append(trans_label)
        is_post = convergence_idx >= 0 and t_idx >= convergence_idx

        # Aggregate born/died across layers from feature_matching
        total_born = 0
        total_died = 0
        for layer in layers:
            fm = feature_matching.get(pair_key, {}).get(layer, {})
            b = fm.get('n_born', 0)
            d = fm.get('n_died', 0)
            total_born += b
            total_died += d
            if is_post:
                layer_post_churn[layer].append(b + d)
            else:
                layer_pre_churn[layer].append(b + d)

        # Get born/died SSI/CSI from transition JSONs
        born_ssis, born_csis = [], []
        died_ssis, died_csis = [], []
        for layer in layers:
            layer_file = transitions_dir / pair_key / f"{_safe_layer_filename(layer)}.json"
            if not layer_file.exists():
                continue
            try:
                with open(layer_file) as f:
                    entries = json.load(f)
                for entry in entries:
                    for born in entry.get('born', []):
                        ssi = born.get('ssi', 0.0)
                        csi = born.get('csi', 0.0)
                        born_ssis.append(ssi)
                        born_csis.append(csi)
                    for died in entry.get('died', []):
                        ssi = died.get('ssi', 0.0)
                        csi = died.get('csi', 0.0)
                        died_ssis.append(ssi)
                        died_csis.append(csi)
            except Exception:
                continue

        b_mean_ssi = float(np.mean(born_ssis)) if born_ssis else 0.0
        b_mean_csi = float(np.mean(born_csis)) if born_csis else 0.0
        d_mean_ssi = float(np.mean(died_ssis)) if died_ssis else 0.0
        d_mean_csi = float(np.mean(died_csis)) if died_csis else 0.0

        if is_post:
            post_born_ssi.extend(born_ssis)
            post_born_csi.extend(born_csis)
            post_died_ssi.extend(died_ssis)
            post_died_csi.extend(died_csis)
        else:
            pre_born_ssi.extend(born_ssis)
            pre_born_csi.extend(born_csis)
            pre_died_ssi.extend(died_ssis)
            pre_died_csi.extend(died_csis)

        val_acc = accuracies[t_idx] if t_idx < len(accuracies) else None
        drift_points.append(DriftTransitionPoint(
            transition=trans_label,
            is_post_convergence=is_post,
            born_count=total_born,
            died_count=total_died,
            churn=total_born + total_died,
            born_mean_ssi=b_mean_ssi,
            born_mean_csi=b_mean_csi,
            died_mean_ssi=d_mean_ssi,
            died_mean_csi=d_mean_csi,
            val_accuracy=val_acc,
        ))

    # Phase 3: Persistence - alive at convergence vs terminal
    convergence_alive = 0
    terminal_survivors = 0
    persistence_frac = 0.0
    convergence_trans = ""

    if convergence_idx >= 0:
        conv_ckpt = checkpoint_labels[convergence_idx]
        convergence_trans = all_transitions[convergence_idx] if convergence_idx < len(all_transitions) else ""

        for layer in layers:
            fl = feature_landscape.get(conv_ckpt, {}).get(layer, {})
            convergence_alive += fl.get('n_alive', 0)

        # Survivors from convergence to terminal: running minimum of stable counts per layer
        surviving_per_layer: dict[str, int] = {}
        for layer in layers:
            fl = feature_landscape.get(conv_ckpt, {}).get(layer, {})
            surviving_per_layer[layer] = fl.get('n_alive', 0)

        for t_idx in range(convergence_idx, n_trans):
            ckpt_a = checkpoint_labels[t_idx]
            ckpt_b = checkpoint_labels[t_idx + 1]
            pair_key = f"{ckpt_a}_to_{ckpt_b}"
            for layer in layers:
                fm = feature_matching.get(pair_key, {}).get(layer, {})
                n_stable = fm.get('n_stable', 0)
                surviving_per_layer[layer] = min(surviving_per_layer[layer], n_stable)

        terminal_survivors = sum(surviving_per_layer.values())
        persistence_frac = terminal_survivors / max(convergence_alive, 1)

    # Phase 4: Per-layer drift ratio
    layer_drift = []
    for layer in layers:
        pre = layer_pre_churn[layer]
        post = layer_post_churn[layer]
        pre_rate = float(np.mean(pre)) if pre else 0.0
        post_rate = float(np.mean(post)) if post else 0.0
        ratio = post_rate / max(pre_rate, 1e-6)
        layer_drift.append(LayerDriftPoint(
            layer=layer,
            pre_convergence_churn_rate=pre_rate,
            post_convergence_churn_rate=post_rate,
            drift_ratio=ratio,
        ))

    # Outcome label
    post_transitions = [dp for dp in drift_points if dp.is_post_convergence]
    pre_transitions = [dp for dp in drift_points if not dp.is_post_convergence]
    pre_mean_churn = float(np.mean([dp.churn for dp in pre_transitions])) if pre_transitions else 0.0
    post_mean_churn = float(np.mean([dp.churn for dp in post_transitions])) if post_transitions else 0.0

    if convergence_idx < 0:
        drift_verdict = "No Convergence Detected"
        has_drift = False
    elif len(post_transitions) < 2:
        drift_verdict = "Insufficient Post-Convergence Data"
        has_drift = False
    elif pre_mean_churn > 0 and post_mean_churn > 0.5 * pre_mean_churn:
        drift_verdict = "Silent Drift"
        has_drift = True
    else:
        drift_verdict = "Stable After Convergence"
        has_drift = False

    return RepresentationalDriftData(
        convergence_transition_idx=convergence_idx,
        convergence_transition=convergence_trans,
        convergence_val_accuracy=convergence_acc,
        convergence_method=convergence_method,
        drift_points=drift_points,
        transitions=all_transitions,
        pre_born_mean_ssi=float(np.mean(pre_born_ssi)) if pre_born_ssi else 0.0,
        pre_born_mean_csi=float(np.mean(pre_born_csi)) if pre_born_csi else 0.0,
        post_born_mean_ssi=float(np.mean(post_born_ssi)) if post_born_ssi else 0.0,
        post_born_mean_csi=float(np.mean(post_born_csi)) if post_born_csi else 0.0,
        pre_died_mean_ssi=float(np.mean(pre_died_ssi)) if pre_died_ssi else 0.0,
        pre_died_mean_csi=float(np.mean(pre_died_csi)) if pre_died_csi else 0.0,
        post_died_mean_ssi=float(np.mean(post_died_ssi)) if post_died_ssi else 0.0,
        post_died_mean_csi=float(np.mean(post_died_csi)) if post_died_csi else 0.0,
        convergence_alive_count=convergence_alive,
        terminal_survivors=terminal_survivors,
        persistence_fraction=persistence_frac,
        layer_drift=layer_drift,
        layers=layers,
        has_post_convergence_drift=has_drift,
        drift_verdict=drift_verdict,
    )


@router.get("/dataset-overview/{stove_id}", response_model=DatasetOverviewResponse)
async def get_dataset_overview(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                               exclude_stable: bool = Query(False, description="Exclude stable features from process counts")):
    """Lightweight dataset overview + process intensity - no heavy computation."""
    try:
        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)
        if not sae_data:
            return DatasetOverviewResponse()

        metadata = sae_data.get('metadata', {})
        layers = metadata.get('layers', [])
        checkpoint_labels = metadata.get('checkpoint_labels', [])
        k_sparse = metadata.get('k_sparse', 32)
        n_classes = metadata.get('n_classes', 0)
        sc_summary = sae_data.get('superclass_summary', {})

        n_samples_per_class = metadata.get('n_samples_per_class', 0)
        # Fallback: derive the count from the activation .npy files
        if n_samples_per_class == 0 and n_classes > 0 and checkpoint_labels and layers:
            try:
                import numpy as np
                act_path = sae_dir / "activations" / checkpoint_labels[0]
                for npy_file in act_path.glob("*.npy"):
                    H = np.load(npy_file, mmap_mode='r')
                    n_samples_per_class = H.shape[0] // n_classes
                    break
            except Exception:
                pass

        dataset_overview = InsightsDatasetOverview(
            n_classes=n_classes,
            n_superclasses=len(sc_summary),
            n_layers=len(layers),
            n_checkpoints=len(checkpoint_labels),
            checkpoint_labels=checkpoint_labels,
            approach=metadata.get('approach', ''),
            expansion_factor=metadata.get('expansion_factor', 4),
            k_sparse=k_sparse,
            n_samples_per_class=n_samples_per_class,
        )

        adj_global: dict[str, dict[str, int]] = {}
        if exclude_stable:
            selectivity = sae_data_raw.get('selectivity', sae_data.get('selectivity', {}))
            adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
            adj_global = adjs.get('global', {})

        pi_list: list[ProcessIntensityPoint] = []
        for pi in sae_data.get('process_intensity', []):
            if not isinstance(pi, dict):
                continue
            t = pi.get('transition', '')
            adj = adj_global.get(t, {}) if exclude_stable else {}
            ab_h = max(0, pi.get('ab_h', 0) - adj.get('ab_h', 0))
            tg_h = max(0, pi.get('tg_h', 0) - adj.get('tg_h', 0))
            di_h = max(0, pi.get('di_h', 0) - adj.get('di_h', 0))
            as_h = pi.get('as_h', 0)
            de_h = pi.get('de_h', 0)
            # tg_h is a subcategory of ab_h and is not added to the total
            total = ab_h + di_h + as_h + de_h + pi.get('unclassified', 0)
            proc_counts = {'ab_h': ab_h, 'di_h': di_h, 'as_h': as_h, 'de_h': de_h}
            dominant = max(proc_counts, key=proc_counts.get) if total > 0 else 'none'  # type: ignore
            pi_list.append(ProcessIntensityPoint(
                transition=t, ab_h=ab_h, tg_h=tg_h, di_h=di_h, as_h=as_h, de_h=de_h,
                total=total, dominant=dominant,
            ))

        # Enrich with training metadata (accuracy, weight updates)
        _enrich_pi_with_training_metadata(pi_list, lane['experiment_id'], lane['id'])

        # Feature lifecycle totals from feature_matching
        total_born = total_died = total_stable = total_transformed = 0
        matching_data = sae_data.get('feature_matching', {})
        for _transition, layer_dict in matching_data.items():
            if not isinstance(layer_dict, dict):
                continue
            for _layer, vals in layer_dict.items():
                if not isinstance(vals, dict):
                    continue
                total_born += vals.get('n_born', 0)
                total_died += vals.get('n_died', 0)
                total_stable += vals.get('n_stable', 0)
                total_transformed += vals.get('n_transformed', 0)

        # Process totals from process_intensity
        total_ab_h = sum(p.ab_h for p in pi_list)
        total_tg_h = sum(p.tg_h for p in pi_list)
        total_di_h = sum(p.di_h for p in pi_list)
        total_as_h = sum(p.as_h for p in pi_list)
        total_de_h = sum(p.de_h for p in pi_list)

        # Per-sample observation totals from transition files
        total_born_obs = total_stable_obs = total_died_obs = 0
        transitions_subdir = "transitions" if matching_method == "activation" else "transitions_weight"
        trans_dir = sae_dir / transitions_subdir
        if trans_dir.exists():
            import json as _json
            for pair_dir in sorted(trans_dir.iterdir()):
                if not pair_dir.is_dir():
                    continue
                for layer_file in sorted(pair_dir.iterdir()):
                    if not layer_file.suffix == '.json':
                        continue
                    with open(layer_file) as f:
                        samples = _json.load(f)
                    for s in samples:
                        total_born_obs += len(s.get('born', []))
                        total_stable_obs += len(s.get('stable', []))
                        total_died_obs += len(s.get('died', []))

        at = metadata.get('analysis_thresholds', {}) or {}
        breakdown = ProcessClassificationBreakdown(
            total_born=total_born,
            total_died=total_died,
            total_stable=total_stable,
            total_transformed=total_transformed,
            total_born_observations=total_born_obs,
            total_stable_observations=total_stable_obs,
            total_died_observations=total_died_obs,
            total_ab_h=total_ab_h,
            total_tg_h=total_tg_h,
            total_di_h=total_di_h,
            total_as_h=total_as_h,
            total_de_h=total_de_h,
            ssi_abstraction_thresh=at.get('ssi_abstraction_thresh', 0.3),
            sai_task_general_thresh=at.get('sai_task_general_thresh', 0.9),
            csi_differentiation_thresh=at.get('csi_differentiation_thresh', 0.4),
            match_stable_thresh=at.get('match_stable_thresh', 0.5),
            match_death_thresh=at.get('match_death_thresh', 0.2),
        )

        return DatasetOverviewResponse(dataset_overview=dataset_overview, process_intensity=pi_list, classification_breakdown=breakdown)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _get_insights_context(sae_data_raw: dict, sae_data: dict, sae_dir, exclude_stable: bool, matching_method: str):
    """Extract common context needed by all insight section computations."""
    metadata = sae_data.get('metadata', {})
    layers = metadata.get('layers', [])
    checkpoint_labels = metadata.get('checkpoint_labels', [])
    selected_classes = metadata.get('selected_classes', [])
    n_classes = len(selected_classes)
    k_sparse = metadata.get('k_sparse', 32)

    sc_summary = sae_data.get('superclass_summary', {})
    class_to_sc = {}
    for sc_name, sc_d in sc_summary.items():
        for cls_name in sc_d.get('per_class_process', {}).keys():
            class_to_sc[cls_name] = sc_name

    adj_per_class: Optional[dict] = None
    if exclude_stable:
        selectivity = sae_data_raw.get('selectivity', sae_data.get('selectivity', {}))
        adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
        adj_per_class = adjs['per_class']

    return {
        'metadata': metadata, 'layers': layers, 'checkpoint_labels': checkpoint_labels,
        'selected_classes': selected_classes, 'n_classes': n_classes, 'k_sparse': k_sparse,
        'sc_summary': sc_summary, 'class_to_sc': class_to_sc, 'adj_per_class': adj_per_class,
    }


def _compute_insight_section(section: str, ctx: dict, sae_data_raw: dict, sae_data: dict,
                             sae_dir, lane: dict, matching_method: str, exclude_stable: bool):
    """Compute a single insight section by name. Returns the Pydantic model or None."""
    layers = ctx['layers']
    checkpoint_labels = ctx['checkpoint_labels']
    selected_classes = ctx['selected_classes']
    n_classes = ctx['n_classes']
    k_sparse = ctx['k_sparse']
    sc_summary = ctx['sc_summary']
    class_to_sc = ctx['class_to_sc']
    adj_per_class = ctx['adj_per_class']
    feature_landscape = sae_data.get('feature_landscape', {})
    co_occurrence = sae_data.get('co_occurrence', {})
    feature_matching = sae_data.get('feature_matching', {})

    if section == 'boundary_crystallization':
        return _compute_boundary_crystallization(
            sae_dir, layers, checkpoint_labels, selected_classes, class_to_sc, n_classes, k_sparse)
    elif section == 'temporal_ordering':
        return _compute_temporal_ordering(sae_data, sc_summary, selected_classes, class_to_sc, adj_per_class)
    elif section == 'feature_genealogy':
        return _compute_feature_genealogy(sae_dir, layers, checkpoint_labels, sae_data, matching_method)
    elif section == 'cross_layer_composition':
        return _compute_cross_layer_composition(sae_dir, layers, checkpoint_labels, n_classes)
    elif section == 'information_bottleneck':
        return _compute_information_bottleneck(sae_dir, layers, checkpoint_labels, feature_landscape, k_sparse)
    elif section == 'superposition_evolution':
        return _compute_superposition_evolution(feature_landscape, co_occurrence, layers, checkpoint_labels, k_sparse)
    elif section == 'forgettable_examples':
        return _compute_forgettable_examples(sae_dir, layers, checkpoint_labels, selected_classes, n_classes, matching_method)
    elif section == 'acquired_equivalence':
        return _compute_acquired_equivalence(sae_dir, layers, checkpoint_labels, selected_classes, class_to_sc, n_classes)
    elif section == 'layer_abstraction_progression':
        return _compute_layer_abstraction_progression(feature_landscape, feature_matching, layers, checkpoint_labels)
    elif section == 'task_general_abstraction':
        return _compute_task_general_abstraction(feature_landscape, feature_matching, sae_data, layers, checkpoint_labels)
    elif section == 'critical_period':
        return _compute_critical_period(sae_data, checkpoint_labels, adj_per_class)
    elif section == 'representational_geometry':
        return _compute_representational_geometry(sae_dir, layers, checkpoint_labels, n_classes)
    elif section == 'process_interference':
        return _compute_process_interference(sae_dir, layers, checkpoint_labels, n_classes, matching_method)
    elif section == 'catastrophic_forgetting':
        return _compute_catastrophic_forgetting(sae_data, selected_classes, adj_per_class)
    elif section == 'polysemanticity_dynamics':
        return _compute_polysemanticity_dynamics(sae_dir, sae_data, layers, checkpoint_labels, matching_method)
    elif section == 'representational_drift':
        # Needs process_intensity list for convergence point detection
        pi_list = _build_process_intensity(sae_data, sae_data_raw, sae_dir, layers, checkpoint_labels,
                                           matching_method, exclude_stable, lane)
        return _compute_representational_drift(sae_dir, sae_data, layers, checkpoint_labels, matching_method, pi_list)
    elif section == 'feature_survival':
        return _compute_feature_survival(sae_dir, layers, checkpoint_labels, sae_data, matching_method)
    elif section == 'dataset_overview':
        sc_names = list(sc_summary.keys())
        n_samples_per_class = ctx['metadata'].get('n_samples_per_class', 0)
        if n_samples_per_class == 0 and n_classes > 0 and checkpoint_labels and layers:
            try:
                import numpy as np
                act_path = sae_dir / "activations" / checkpoint_labels[0]
                for npy_file in act_path.glob("*.npy"):
                    H = np.load(npy_file, mmap_mode='r')
                    n_samples_per_class = H.shape[0] // n_classes
                    break
            except Exception:
                pass
        return InsightsDatasetOverview(
            n_classes=n_classes, n_superclasses=len(sc_names), n_layers=len(layers),
            n_checkpoints=len(checkpoint_labels), checkpoint_labels=checkpoint_labels,
            approach=ctx['metadata'].get('approach', ''), expansion_factor=ctx['metadata'].get('expansion_factor', 4),
            k_sparse=k_sparse, n_samples_per_class=n_samples_per_class,
        )
    elif section == 'process_intensity':
        return _build_process_intensity(sae_data, sae_data_raw, sae_dir, layers, checkpoint_labels,
                                        matching_method, exclude_stable, lane)
    else:
        return None


def _build_process_intensity(sae_data: dict, sae_data_raw: dict, sae_dir, layers, checkpoint_labels,
                             matching_method: str, exclude_stable: bool, lane: dict) -> list:
    """Build process intensity list (shared by full insights and per-section endpoint)."""
    adj_global = {}
    if exclude_stable:
        selectivity = sae_data_raw.get('selectivity', sae_data.get('selectivity', {}))
        adjs = _compute_stable_adjustments(sae_dir, layers, checkpoint_labels, matching_method, selectivity)
        adj_global = adjs.get('global', {})

    pi_list: list[ProcessIntensityPoint] = []
    for pi in sae_data.get('process_intensity', []):
        if not isinstance(pi, dict):
            continue
        t = pi.get('transition', '')
        adj = adj_global.get(t, {}) if exclude_stable else {}
        ab_h = max(0, pi.get('ab_h', 0) - adj.get('ab_h', 0))
        tg_h = max(0, pi.get('tg_h', 0) - adj.get('tg_h', 0))
        di_h = max(0, pi.get('di_h', 0) - adj.get('di_h', 0))
        as_h = pi.get('as_h', 0)
        de_h = pi.get('de_h', 0)
        total = ab_h + di_h + as_h + de_h + pi.get('unclassified', 0)
        proc_counts = {'ab_h': ab_h, 'di_h': di_h, 'as_h': as_h, 'de_h': de_h}
        dominant = max(proc_counts, key=proc_counts.get) if total > 0 else 'none'  # type: ignore
        pi_list.append(ProcessIntensityPoint(
            transition=t, ab_h=ab_h, tg_h=tg_h, di_h=di_h, as_h=as_h, de_h=de_h,
            total=total, dominant=dominant,
        ))
    _enrich_pi_with_training_metadata(pi_list, lane['experiment_id'], lane['id'])
    return pi_list


# Per-section insights endpoint (progressive loading)

INSIGHT_SECTIONS = [
    'dataset_overview', 'process_intensity',
    'temporal_ordering', 'critical_period', 'process_interference',
    'boundary_crystallization', 'acquired_equivalence', 'representational_geometry',
    'layer_abstraction_progression', 'task_general_abstraction',
    'feature_genealogy', 'cross_layer_composition', 'superposition_evolution',
    'information_bottleneck', 'polysemanticity_dynamics', 'feature_survival',
    'forgettable_examples', 'catastrophic_forgetting', 'representational_drift',
]


@router.get("/insights/{stove_id}/section/{section_name}")
async def get_insight_section(stove_id: str, section_name: str,
                              matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                              exclude_stable: bool = Query(False)):
    """Compute and return a single insight section for progressive loading."""
    if section_name not in INSIGHT_SECTIONS:
        raise HTTPException(status_code=400, detail=f"Unknown section: {section_name}")
    try:
        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)
        if not sae_data:
            return JSONResponse(content=None)

        ctx = _get_insights_context(sae_data_raw, sae_data, sae_dir, exclude_stable, matching_method)
        result = _compute_insight_section(
            section_name, ctx, sae_data_raw, sae_data, sae_dir, lane, matching_method, exclude_stable)
        if result is None:
            return JSONResponse(content=None)
        if isinstance(result, list):
            return JSONResponse(content=_sanitize_for_json([item.dict() if hasattr(item, 'dict') else item for item in result]))
        if hasattr(result, 'dict'):
            return JSONResponse(content=_sanitize_for_json(result.dict()))
        return JSONResponse(content=_sanitize_for_json(result))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing insight section {section_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/{stove_id}/sections-list")
async def get_insight_sections_list():
    """Return the list of available insight section names."""
    return JSONResponse(content=INSIGHT_SECTIONS)


@router.get("/insights/{stove_id}", response_model=InsightsResponse)
async def get_insights(stove_id: str, matching_method: str = Query("activation", pattern="^(activation|weight)$"),
                       exclude_stable: bool = Query(False, description="Exclude stable features from process counts")):
    """Scientific deep-dive analyses - returns all sections at once (used for PDF export)."""
    try:
        stove, lane, experiment, sae_dir, sae_data_raw = await _load_sae_context(stove_id)
        sae_data = _resolve_method_data(sae_data_raw, matching_method)

        if not sae_data:
            return InsightsResponse()

        ctx = _get_insights_context(sae_data_raw, sae_data, sae_dir, exclude_stable, matching_method)

        results = {}
        for section in INSIGHT_SECTIONS:
            results[section] = _compute_insight_section(
                section, ctx, sae_data_raw, sae_data, sae_dir, lane, matching_method, exclude_stable)

        response = InsightsResponse(
            boundary_crystallization=results.get('boundary_crystallization') or BoundaryCrystallizationData(),
            temporal_ordering=results.get('temporal_ordering') or TemporalOrderingData(),
            feature_genealogy=results.get('feature_genealogy') or FeatureGenealogyData(),
            cross_layer_composition=results.get('cross_layer_composition') or CrossLayerCompositionData(),
            information_bottleneck=results.get('information_bottleneck'),
            superposition_evolution=results.get('superposition_evolution'),
            forgettable_examples=results.get('forgettable_examples'),
            acquired_equivalence=results.get('acquired_equivalence'),
            layer_abstraction_progression=results.get('layer_abstraction_progression'),
            task_general_abstraction=results.get('task_general_abstraction'),
            critical_period=results.get('critical_period'),
            representational_geometry=results.get('representational_geometry'),
            process_interference=results.get('process_interference'),
            catastrophic_forgetting=results.get('catastrophic_forgetting'),
            polysemanticity_dynamics=results.get('polysemanticity_dynamics'),
            representational_drift=results.get('representational_drift'),
            feature_survival=results.get('feature_survival'),
            dataset_overview=results.get('dataset_overview') or InsightsDatasetOverview(),
            process_intensity=results.get('process_intensity') or [],
        )
        return JSONResponse(content=_sanitize_for_json(response.dict()))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing insights: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# Start / Reset / Background

@router.post("/start", response_model=StartSAEAnalysisResponse)
async def start_analysis(request: StartSAEAnalysisRequest, background_tasks: BackgroundTasks):
    """Start SAEANALYSIS for a job"""
    try:
        stove_id = request.stove_id
        stove = await StoveDB.get_by_id(stove_id)

        if not stove:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {stove_id} not found")

        if stove['stove_type'].upper() != 'SAEANALYSIS':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job {stove_id} is not a SAEANALYSIS job")

        # DEVTRAIN must be completed
        lane_id = stove['lane_id']
        stoves_in_lane = await StoveDB.get_all(lane_id)

        devtrain = next((s for s in stoves_in_lane if s['stove_type'].upper() == 'DEVTRAIN'), None)

        if not devtrain or devtrain['status'] != 'completed':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="DEVTRAIN must be completed before starting SAEANALYSIS")

        await StoveDB.update_status(stove_id, status="running", progress=0.0)
        background_tasks.add_task(run_saeanalysis_background, stove_id)

        return StartSAEAnalysisResponse(status="started", message="SAEANALYSIS started successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting SAEANALYSIS: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/reset", response_model=ResetSAEAnalysisResponse)
async def reset_analysis(request: ResetSAEAnalysisRequest):
    """Reset SAEANALYSIS job"""
    try:
        stove_id = request.stove_id
        stove = await StoveDB.get_by_id(stove_id)

        if not stove:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {stove_id} not found")

        lane_id = stove['lane_id']
        lane = await LaneDB.get_by_id(lane_id)
        experiment_id = lane['experiment_id']

        sae_dir = get_saeanalysis_dir(experiment_id, lane_id)
        if sae_dir.exists():
            shutil.rmtree(sae_dir)
            logger.info(f"Deleted sae_analysis directory: {sae_dir}")

        await StoveDB.update_status(stove_id, status="ready", progress=0.0)

        return ResetSAEAnalysisResponse(status="reset", message="SAEANALYSIS reset successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting SAEANALYSIS: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/reclassify", response_model=ReclassifySAEAnalysisResponse)
async def reclassify_analysis(request: ReclassifySAEAnalysisRequest, background_tasks: BackgroundTasks):
    """Re-run process classification with new thresholds using saved SAEs & activations."""
    try:
        stove_id = request.stove_id
        stove = await StoveDB.get_by_id(stove_id)

        if not stove:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {stove_id} not found")

        if stove['stove_type'].upper() != 'SAEANALYSIS':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Job {stove_id} is not a SAEANALYSIS job")

        if stove['status'] != 'completed':
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="SAEANALYSIS must be completed before reclassifying")

        await StoveDB.update_status(stove_id, status="running", progress=0.0)
        background_tasks.add_task(
            run_reclassify_background, stove_id, request.analysis_thresholds
        )

        return ReclassifySAEAnalysisResponse(
            status="started",
            message="Reclassification started; reusing saved SAEs and activations"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting reclassification: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


async def run_reclassify_background(stove_id: str, analysis_thresholds: Optional[dict] = None):
    """Background task to run reclassification with new thresholds."""
    polling_active = True
    poll_task = None
    try:
        stove = await StoveDB.get_by_id(stove_id)
        lane = await LaneDB.get_by_id(stove['lane_id'])
        experiment = await ExperimentDB.get_by_id(lane['experiment_id'])

        experiment_id = experiment['id']
        lane_id = lane['id']
        selected_classes = experiment['selected_classes']

        lane_dir = get_lane_dir(experiment_id, lane_id)

        logger.info(f"Starting reclassification for job {stove_id}")

        await ConnectionManager.broadcast(
            stove_id,
            SAEAnalysisWebSocketMessage(
                message_type="progress",
                stove_id=stove_id,
                progress=0.0,
                current_step="Starting reclassification"
            )
        )

        import sys
        ml_package_path = Path(__file__).parent.parent.parent.parent / "ml" / "src"
        if str(ml_package_path) not in sys.path:
            sys.path.insert(0, str(ml_package_path))

        from saeanalysis import reclassify_from_saved

        loop = asyncio.get_event_loop()
        progress_file_path = lane_dir / "sae_analysis" / "saeanalysis_progress.json"

        # Same experiment config as start_analysis
        experiment_config = {
            'dataset_id': experiment['dataset_id'],
            'selected_classes': selected_classes,
            'experiment_id': experiment_id,
        }

        stoves_in_lane = await StoveDB.get_all(lane_id)
        netinit = next((s for s in stoves_in_lane if s['stove_type'].upper() == 'NETINIT'), None)
        if netinit and netinit.get('configuration'):
            config_data = netinit['configuration']
            if isinstance(config_data, str):
                config_data = json.loads(config_data)
            experiment_config['netinit'] = config_data
            for key in ('sae_policy', 'transform_config', 'memory_policy', 'validation_policy'):
                if key in config_data:
                    experiment_config[key] = config_data[key]

        async def poll_progress():
            last_pct = -1.0
            while polling_active:
                await asyncio.sleep(2)
                try:
                    if progress_file_path.exists():
                        data = json.loads(progress_file_path.read_text())
                        pct = data.get("progress_pct", 0)
                        msg = data.get("message", "")
                        if pct != last_pct:
                            last_pct = pct
                            progress_normalized = pct / 100.0
                            await StoveDB.update_status(stove_id, status="running", progress=progress_normalized)
                            await ConnectionManager.broadcast(
                                stove_id,
                                SAEAnalysisWebSocketMessage(
                                    message_type="progress",
                                    stove_id=stove_id,
                                    progress=progress_normalized,
                                    current_step=msg
                                )
                            )
                except Exception:
                    pass

        poll_task = asyncio.create_task(poll_progress())

        import functools
        results = await loop.run_in_executor(
            None,
            functools.partial(
                reclassify_from_saved,
                str(lane_dir),
                selected_classes,
                experiment_config,
                None,
                progress_file=str(progress_file_path),
                analysis_thresholds=analysis_thresholds,
            )
        )

        polling_active = False
        poll_task.cancel()

        for _attempt in range(5):
            try:
                await StoveDB.update_status(stove_id, status="completed", progress=1.0)
                break
            except Exception as db_err:
                if "locked" in str(db_err) and _attempt < 4:
                    logger.warning(f"DB locked on reclassification completion, retry {_attempt + 1}/5")
                    await asyncio.sleep(2 ** _attempt)
                else:
                    raise

        await ConnectionManager.broadcast(
            stove_id,
            SAEAnalysisWebSocketMessage(
                message_type="analysis_complete",
                stove_id=stove_id,
                progress=1.0,
                current_step="Reclassification complete"
            )
        )

        logger.info(f"Reclassification completed for job {stove_id}")

    except Exception as e:
        logger.error(f"Reclassification failed for job {stove_id}: {e}", exc_info=True)
        polling_active = False
        if poll_task:
            poll_task.cancel()
        for _attempt in range(5):
            try:
                await StoveDB.update_status(stove_id, status="error", progress=0.0)
                break
            except Exception:
                if _attempt < 4:
                    await asyncio.sleep(2 ** _attempt)
                else:
                    logger.error(f"Failed to update job {stove_id} to error status after 5 retries")
        await ConnectionManager.broadcast(
            stove_id,
            SAEAnalysisWebSocketMessage(
                message_type="error",
                stove_id=stove_id,
                progress=0.0,
                current_step=f"Reclassification failed: {str(e)}"
            )
        )


async def run_saeanalysis_background(stove_id: str):
    """Background task to run SAEANALYSIS"""
    polling_active = True
    poll_task = None
    try:
        stove = await StoveDB.get_by_id(stove_id)
        lane = await LaneDB.get_by_id(stove['lane_id'])
        experiment = await ExperimentDB.get_by_id(lane['experiment_id'])

        experiment_id = experiment['id']
        lane_id = lane['id']
        selected_classes = experiment['selected_classes']

        lane_dir = get_lane_dir(experiment_id, lane_id)
        sae_dir = get_saeanalysis_dir(experiment_id, lane_id)
        sae_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting SAEANALYSIS for job {stove_id}")

        await ConnectionManager.broadcast(
            stove_id,
            SAEAnalysisWebSocketMessage(
                message_type="progress",
                stove_id=stove_id,
                progress=0.0,
                current_step="Starting SAE analysis"
            )
        )

        import sys
        ml_package_path = Path(__file__).parent.parent.parent.parent / "ml" / "src"
        if str(ml_package_path) not in sys.path:
            sys.path.insert(0, str(ml_package_path))

        from saeanalysis import analyze_sae_features

        loop = asyncio.get_event_loop()
        progress_file_path = lane_dir / "sae_analysis" / "saeanalysis_progress.json"
        progress_file_path.parent.mkdir(parents=True, exist_ok=True)

        async def poll_progress():
            last_pct = -1.0
            while polling_active:
                await asyncio.sleep(2)
                try:
                    if progress_file_path.exists():
                        data = json.loads(progress_file_path.read_text())
                        pct = data.get("progress_pct", 0)
                        msg = data.get("message", "")
                        if pct != last_pct:
                            last_pct = pct
                            progress_normalized = pct / 100.0
                            await StoveDB.update_status(stove_id, status="running", progress=progress_normalized)
                            await ConnectionManager.broadcast(
                                stove_id,
                                SAEAnalysisWebSocketMessage(
                                    message_type="progress",
                                    stove_id=stove_id,
                                    progress=progress_normalized,
                                    current_step=msg
                                )
                            )
                except Exception:
                    pass

        poll_task = asyncio.create_task(poll_progress())

        experiment_config = {
            'dataset_id': experiment['dataset_id'],
            'selected_classes': selected_classes,
            'experiment_id': experiment_id,
        }

        stoves_in_lane = await StoveDB.get_all(lane_id)
        netinit = next((s for s in stoves_in_lane if s['stove_type'].upper() == 'NETINIT'), None)
        if netinit and netinit.get('configuration'):
            config_data = netinit['configuration']
            if isinstance(config_data, str):
                config_data = json.loads(config_data)
            experiment_config['netinit'] = config_data
            for key in ('sae_policy', 'transform_config', 'memory_policy', 'validation_policy'):
                if key in config_data:
                    experiment_config[key] = config_data[key]

        # Extract analysis thresholds from sae_policy if present
        sae_policy = experiment_config.get('sae_policy', {}) or {}
        analysis_thresholds = sae_policy.get('analysis_thresholds', None)

        logger.info(f"Launching analyze_sae_features in executor: lane_dir={lane_dir}, classes={len(selected_classes)}")

        import functools
        results = await loop.run_in_executor(
            None,
            functools.partial(
                analyze_sae_features,
                str(lane_dir),
                selected_classes,
                experiment_config,
                None,
                progress_file=str(progress_file_path),
                analysis_thresholds=analysis_thresholds,
            )
        )

        polling_active = False
        poll_task.cancel()

        # Retry on transient SQLite "database is locked" errors
        for _attempt in range(5):
            try:
                await StoveDB.update_status(stove_id, status="completed", progress=1.0)
                break
            except Exception as db_err:
                if "locked" in str(db_err) and _attempt < 4:
                    logger.warning(f"DB locked on completion update, retry {_attempt + 1}/5")
                    await asyncio.sleep(2 ** _attempt)
                else:
                    raise

        await ConnectionManager.broadcast(
            stove_id,
            SAEAnalysisWebSocketMessage(
                message_type="analysis_complete",
                stove_id=stove_id,
                progress=1.0,
                current_step="SAE analysis complete"
            )
        )

        logger.info(f"SAEANALYSIS completed for job {stove_id}")

    except Exception as e:
        import traceback
        polling_active = False
        if poll_task:
            poll_task.cancel()
        logger.error(f"Error in SAEANALYSIS background task: {e}", exc_info=True)
        traceback.print_exc()
        for _attempt in range(5):
            try:
                await StoveDB.update_status(stove_id, status="error", progress=0.0)
                break
            except Exception:
                if _attempt < 4:
                    await asyncio.sleep(2 ** _attempt)
                else:
                    logger.error(f"Failed to update job {stove_id} to error status after 5 retries")
        await ConnectionManager.broadcast(
            stove_id,
            SAEAnalysisWebSocketMessage(
                message_type="error",
                stove_id=stove_id,
                error_message=str(e)
            )
        )
