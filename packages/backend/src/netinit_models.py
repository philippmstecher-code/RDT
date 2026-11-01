"""
Network Initialization (NETINIT) stove models and data
"""
from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field
from enum import Enum


class NetworkType(str, Enum):
    RESNET18 = "resnet18"
    RESNET34 = "resnet34"
    RESNET50 = "resnet50"
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    VIT_BASE = "vit_base"
    VIT_SMALL = "vit_small"
    VIT_TINY = "vit_tiny"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    DENSENET121 = "densenet121"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    INCEPTION_V3 = "inception_v3"
    SWIN_TINY = "swin_tiny"
    SWIN_SMALL = "swin_small"
    MAXVIT_T = "maxvit_t"
    CCT_7 = "cct_7"
    UNET = "unet"


class InitializationMethod(str, Enum):
    RANDOM_NORMAL = "random_normal"
    RANDOM_UNIFORM = "random_uniform"
    XAVIER_NORMAL = "xavier_normal"
    XAVIER_UNIFORM = "xavier_uniform"
    KAIMING_NORMAL = "kaiming_normal"
    KAIMING_UNIFORM = "kaiming_uniform"
    PRETRAINED_IMAGENET = "pretrained_imagenet"
    ZEROS = "zeros"
    ONES = "ones"


class NetworkArchitecture(BaseModel):
    """Available network architecture with metadata"""
    id: NetworkType
    name: str
    description: str
    parameter_count: int
    typical_training_time_per_epoch_seconds: float


class InitializationOption(BaseModel):
    """Available weight initialization method"""
    id: InitializationMethod
    name: str
    description: str
    requires_pretrained: bool


class DistributionScheme(str, Enum):
    UNIFORM = "uniform"
    DENSE_EARLY = "dense_early"
    DENSE_LATE = "dense_late"


class TerminalCaptureMode(str, Enum):
    AT_THRESHOLD = "at_threshold"
    FINAL_EPOCH = "final_epoch"
    BOTH = "both"


class MilestoneType(str, Enum):
    ACCURACY = "accuracy"
    WEIGHT_UPDATES = "weight_updates"


class OptimizerType(str, Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"


class RepresentationMethod(str, Enum):
    """Methods for operationalizing neural network representations"""
    MEAN_ACTIVATIONS = "mean_activations"
    INDIVIDUAL_ACTIVATIONS = "individual_activations"
    LINEAR_PROBING = "linear_probing"
    MULTILAYER_ACTIVATIONS = "multilayer_activations"
    INDIVIDUAL_MULTILAYER_ACTIVATIONS = "individual_multilayer_activations"


class TrainingPolicy(BaseModel):
    """
    Training stopping criteria with OR logic.
    Training stops when EITHER condition is met (whichever comes first).
    At least one criterion must be specified.
    """
    epochs: Optional[int] = Field(None, ge=1, description="Maximum number of training epochs")
    accuracy_threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Target accuracy percentage (0-100). Training stops when reached."
    )

    def model_post_init(self, __context) -> None:
        """Validate that at least one stopping criterion is provided."""
        if self.epochs is None and self.accuracy_threshold is None:
            raise ValueError("At least one stopping criterion (epochs or accuracy_threshold) must be specified")


class SnapshotPolicy(BaseModel):
    """Snapshot policy for capturing model states during training"""
    milestone_count: int = Field(ge=2, le=50, description="Number of snapshots to take", alias="snapshot_count")
    distribution_scheme: DistributionScheme = Field(
        default=DistributionScheme.UNIFORM,
        description="How to distribute snapshots across training"
    )
    samples_per_class: Optional[int] = Field(
        300,
        ge=1,
        description="Number of samples per class for activation extraction (None = all samples)"
    )
    terminal_capture: TerminalCaptureMode = Field(
        default=TerminalCaptureMode.BOTH,
        description="When to capture the terminal snapshot"
    )
    milestone_accuracies: Optional[List[float]] = Field(
        None,
        description="Planned accuracy milestones (%) calculated during NETINIT - must match milestone_count"
    )
    milestone_type: MilestoneType = Field(
        default=MilestoneType.ACCURACY,
        description="Milestone trigger mode: 'accuracy' captures at accuracy thresholds, "
                    "'weight_updates' captures at specific batch/gradient-step counts"
    )
    milestone_weight_updates: Optional[List[int]] = Field(
        None,
        description="Planned weight-update milestones (absolute batch counts). "
                    "Used when milestone_type='weight_updates'. "
                    "If None, computed at runtime from epochs * batches_per_epoch."
    )

    class Config:
        populate_by_name = True  # Allow both milestone_count and snapshot_count


class ValidationPolicy(BaseModel):
    """
    Validation granularity controls for intra-epoch validation checks.
    Controls how often validation runs during training to detect milestones.
    """
    frequency_mode: Literal["percentage", "batches"] = Field(
        default="percentage",
        description="How to measure validation frequency"
    )
    frequency_value: float = Field(
        default=5.0,
        ge=0.1,
        le=100.0,
        description="Percentage of epoch (1-100) or number of batches (1+) between validations"
    )
    min_batch_interval: int = Field(
        default=10,
        ge=1,
        description="Minimum batches between validation checks (safety floor)"
    )
    validation_set_fraction: float = Field(
        default=1.0,
        ge=0.01,
        le=1.0,
        description="Fraction of validation set to use per check (1.0 = full set)"
    )

    def get_check_interval(self, total_batches: int) -> int:
        """
        Calculate batch interval for validation checks.

        Args:
            total_batches: Total batches in one epoch

        Returns:
            Number of batches between validation checks
        """
        if self.frequency_mode == "percentage":
            # Convert percentage to batch interval
            interval = max(1, int(total_batches * (self.frequency_value / 100.0)))
        else:
            # Direct batch count
            interval = max(1, int(self.frequency_value))

        # Apply minimum interval floor
        return max(interval, self.min_batch_interval)


class MemoryPolicy(BaseModel):
    """
    Memory optimization controls for training process.
    Determines which features use disk storage vs RAM accumulation.
    """
    disk_based_linear_probes: bool = Field(
        default=True,
        description="Store activations to disk during linear probe extraction (reduces RAM by ~30GB)"
    )
    disk_based_multilayer: bool = Field(
        default=True,
        description="Stream multilayer activations to disk (reduces RAM by ~30GB)"
    )
    linear_probe_batch_size: int = Field(
        default=64,
        ge=1,
        le=512,
        description="Batch size for disk-based linear probe extraction (higher = faster but more RAM)"
    )
    activation_extraction_enabled: bool = Field(
        default=True,
        description="Enable periodic activation extraction during training (disable to save time/RAM)"
    )
    activation_extraction_interval: int = Field(
        default=20,
        ge=1,
        description="Extract activations every N batches during training"
    )


class LoggingPolicy(BaseModel):
    """
    Logging granularity controls for batch-level and metrics logging.
    Controls how much detail is logged and how often metrics are persisted.
    """
    batch_log_frequency: int = Field(
        default=1,
        ge=1,
        description="Log every Nth batch (1 = all batches, 10 = every 10th batch)"
    )
    batch_log_include_activations: bool = Field(
        default=True,
        description="Include activation statistics in batch log (adds ~1KB per logged batch)"
    )
    metrics_flush_interval: int = Field(
        default=10,
        ge=1,
        description="Flush metrics history to disk every N epochs (prevents unbounded RAM growth)"
    )
    metrics_keep_recent: int = Field(
        default=100,
        ge=10,
        description="Keep last N epochs in RAM before flushing older metrics to disk"
    )


class SAEPolicy(BaseModel):
    """
    Sparse Autoencoder analysis configuration.
    Controls which milestones get their own SAE dictionaries for
    cross-temporal feature matching and process identification.
    """
    sae_milestones: Optional[List[int]] = Field(
        None,
        description="Milestone indices at which to train separate SAEs. "
                    "If None, only terminal SAE is trained (current behavior). "
                    "Example: [0, 3, 7] trains SAEs at milestones 0, 3, 7 plus terminal."
    )
    expansion_factor: int = Field(
        4, ge=2, le=16,
        description="SAE hidden dimension multiplier (d_hidden = d_input * expansion_factor)"
    )
    k_sparse: int = Field(
        32, ge=4, le=256,
        description="Top-k sparsity constraint (number of active features per sample)"
    )
    n_steps: int = Field(
        10_000, ge=1000, le=100_000,
        description="Number of SAE training gradient steps per layer"
    )
    shared_init_seed: Optional[int] = Field(
        42,
        description="Seed for shared SAE initialization across milestones. "
                    "All milestones start from the same random weights per layer, "
                    "enabling meaningful decoder-direction matching. Set to None to disable."
    )
    null_permutations: int = Field(
        100, ge=10, le=1000,
        description="Number of label-shuffle permutations for null baseline test. "
                    "Tests whether detected process signals exceed what random label "
                    "assignment would produce."
    )


class OptimizerConfig(BaseModel):
    """Optimizer configuration"""
    type: OptimizerType
    # Common parameters
    learning_rate: float = Field(ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    # SGD-specific
    momentum: Optional[float] = Field(None, ge=0.0, le=1.0)
    nesterov: Optional[bool] = None
    # Adam/AdamW-specific
    beta1: Optional[float] = Field(None, ge=0.0, lt=1.0)
    beta2: Optional[float] = Field(None, ge=0.0, lt=1.0)
    epsilon: Optional[float] = Field(None, gt=0.0)


class InputTransformConfig(BaseModel):
    """Input image transform pipeline configuration.

    Controls how raw dataset images are preprocessed before feeding into the network.
    If not set, a default transform based on dataset type is used.
    """
    resize: Optional[int] = Field(
        None,
        ge=32,
        description="Resize shorter edge to this size before cropping. None = no resize."
    )
    center_crop: Optional[int] = Field(
        None,
        ge=32,
        description="Center-crop to this square size after resize. None = no crop."
    )
    normalize_mean: List[float] = Field(
        default=[0.5, 0.5, 0.5],
        description="Per-channel normalization mean (RGB). CIFAR: [0.5,0.5,0.5], ImageNet: [0.485,0.456,0.406]"
    )
    normalize_std: List[float] = Field(
        default=[0.5, 0.5, 0.5],
        description="Per-channel normalization std (RGB). CIFAR: [0.5,0.5,0.5], ImageNet: [0.229,0.224,0.225]"
    )
    # Data augmentation (applied to training set only)
    augmentation_enabled: bool = Field(
        default=True,
        description="Enable data augmentation for training set"
    )
    random_crop_padding: Optional[int] = Field(
        default=4,
        ge=0,
        le=16,
        description="Padding for RandomCrop. Set to 0 or None to disable."
    )
    horizontal_flip: bool = Field(
        default=True,
        description="Apply RandomHorizontalFlip during training"
    )
    color_jitter_brightness: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ColorJitter brightness factor (None = disabled)"
    )
    color_jitter_contrast: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ColorJitter contrast factor (None = disabled)"
    )
    color_jitter_saturation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="ColorJitter saturation factor (None = disabled)"
    )


class CurriculumPhase(BaseModel):
    """A single phase in a curriculum training schedule."""
    start_epoch: int = Field(ge=1, description="First epoch of this phase (1-based, inclusive)")
    end_epoch: int = Field(ge=1, description="Last epoch of this phase (1-based, inclusive)")
    label_mode: str = Field(
        description="Label mode: 'superclass' remaps fine-class labels to superclass indices, "
                    "'fine' uses original fine-class labels"
    )
    learning_rate: Optional[float] = Field(
        default=None, ge=0.0,
        description="Override learning rate for this phase (None = keep current)"
    )


class CurriculumPolicy(BaseModel):
    """
    Curriculum training policy: train with different label targets across phases.

    When enabled, DEVTRAIN switches the loss target (superclass vs fine-class labels)
    and optionally the model head at phase boundaries. The model backbone is preserved
    across phases; only the classifier head is replaced when num_classes changes.
    """
    enabled: bool = Field(default=False, description="Enable curriculum phase switching")
    phases: List[CurriculumPhase] = Field(
        default_factory=list,
        description="Ordered list of training phases. Must cover all epochs without gaps."
    )


class NoisePolicy(BaseModel):
    """
    Label noise policy for targeted noise experiments.
    When enabled, training labels are stochastically corrupted according to noise_type.
    """
    enabled: bool = Field(default=False, description="Enable label noise during training")
    noise_type: str = Field(
        default="random",
        description="Noise type: 'within_sc' (same superclass), 'between_sc' (different superclass), 'random' (any class)"
    )
    noise_prob: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability of flipping each label (0.0–1.0)"
    )
    fixed: bool = Field(
        default=False,
        description="If True, corruption is static (drawn once per sample, identical every epoch): "
                    "a memorisable hard corruption. If False, re-randomised each epoch (soft, averaged out)."
    )
    noise_seed: Optional[int] = Field(
        default=None,
        description="RNG seed for the deterministic per-sample draw when fixed=True. "
                    "If None, falls back to the run seed."
    )
    noise_until_epoch: Optional[int] = Field(
        default=None,
        description="Label-noise curriculum gate: noise is active for epochs [start, noise_until_epoch) "
                    "and switched OFF (clean labels) from this epoch onward. None = noise throughout."
    )


class NetInitConfiguration(BaseModel):
    """Complete configuration for NETINIT stove"""
    network_type: NetworkType
    initialization_method: InitializationMethod
    training_policy: TrainingPolicy
    snapshot_policy: SnapshotPolicy
    optimizer: OptimizerConfig
    batch_size: int = Field(ge=1, le=512)
    random_seed: Optional[int] = None
    representation_methods: List[RepresentationMethod] = Field(
        default=[RepresentationMethod.MEAN_ACTIVATIONS],
        description="Methods for operationalizing representations"
    )

    # Resource optimization policies
    validation_policy: ValidationPolicy = Field(
        default_factory=ValidationPolicy,
        description="Validation frequency and set size controls"
    )
    memory_policy: MemoryPolicy = Field(
        default_factory=MemoryPolicy,
        description="Memory optimization controls for disk-based storage"
    )
    logging_policy: LoggingPolicy = Field(
        default_factory=LoggingPolicy,
        description="Logging granularity and metrics persistence controls"
    )

    # SAE analysis policy
    sae_policy: SAEPolicy = Field(
        default_factory=SAEPolicy,
        description="Sparse Autoencoder analysis configuration for multi-milestone feature tracking"
    )

    # Input transform pipeline
    transform_config: Optional[InputTransformConfig] = Field(
        default=None,
        description="Input image transform pipeline. If None, uses the dataset's default transform."
    )

    # Curriculum training policy
    curriculum_policy: Optional[CurriculumPolicy] = Field(
        default=None,
        description="Curriculum training schedule. When enabled, switches label targets across phases."
    )

    # Label noise policy
    noise_policy: Optional[NoisePolicy] = Field(
        default=None,
        description="Label noise policy for targeted noise experiments."
    )


class HardwareRecommendations(BaseModel):
    """Hardware-aware recommended settings for training configuration."""
    disk_based_linear_probes: bool
    disk_based_multilayer: bool
    linear_probe_batch_size: int
    samples_per_class: Optional[int]
    milestone_count: int
    validation_set_fraction: float
    batch_size: int
    rationale: str


class HardwareSpecs(BaseModel):
    """System hardware specifications"""
    cpu_count: int
    cpu_model: str
    total_memory_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_models: List[str]
    gpu_memory_gb: List[float]
    mps_available: bool = False
    device_type: Literal["cuda", "mps", "cpu"] = "cpu"
    recommended_settings: Optional[HardwareRecommendations] = None


class TimeEstimate(BaseModel):
    """Estimated training time and resource requirements"""
    training_time_seconds: float
    total_snapshots: int
    storage_required_gb: float
    estimated_completion_hours: float


# Available network architectures
NETWORK_ARCHITECTURES: list[NetworkArchitecture] = [
    NetworkArchitecture(
        id=NetworkType.RESNET18,
        name="ResNet-18",
        description="18-layer residual network, good balance of speed and accuracy",
        parameter_count=11_689_512,
        typical_training_time_per_epoch_seconds=30,
    ),
    NetworkArchitecture(
        id=NetworkType.RESNET34,
        name="ResNet-34",
        description="34-layer residual network, deeper than ResNet-18",
        parameter_count=21_797_672,
        typical_training_time_per_epoch_seconds=50,
    ),
    NetworkArchitecture(
        id=NetworkType.RESNET50,
        name="ResNet-50",
        description="50-layer residual network with bottleneck blocks",
        parameter_count=25_557_032,
        typical_training_time_per_epoch_seconds=70,
    ),
    NetworkArchitecture(
        id=NetworkType.VGG16,
        name="VGG-16",
        description="16-layer network with small 3x3 convolutions",
        parameter_count=138_357_544,
        typical_training_time_per_epoch_seconds=90,
    ),
    NetworkArchitecture(
        id=NetworkType.VGG19,
        name="VGG-19",
        description="19-layer network, deeper variant of VGG-16",
        parameter_count=143_667_240,
        typical_training_time_per_epoch_seconds=100,
    ),
    NetworkArchitecture(
        id=NetworkType.VIT_BASE,
        name="Vision Transformer Base",
        description="Transformer architecture for computer vision (ViT-B/16)",
        parameter_count=86_567_656,
        typical_training_time_per_epoch_seconds=120,
    ),
    NetworkArchitecture(
        id=NetworkType.VIT_SMALL,
        name="Vision Transformer Small",
        description="Smaller transformer variant, faster training (ViT-S/16, 384-dim)",
        parameter_count=21_704_164,
        typical_training_time_per_epoch_seconds=60,
    ),
    NetworkArchitecture(
        id=NetworkType.EFFICIENTNET_B0,
        name="EfficientNet-B0",
        description="Efficient network with compound scaling, baseline variant",
        parameter_count=5_288_548,
        typical_training_time_per_epoch_seconds=40,
    ),
    NetworkArchitecture(
        id=NetworkType.EFFICIENTNET_B1,
        name="EfficientNet-B1",
        description="Slightly larger EfficientNet variant",
        parameter_count=7_794_184,
        typical_training_time_per_epoch_seconds=50,
    ),
    NetworkArchitecture(
        id=NetworkType.DENSENET121,
        name="DenseNet-121",
        description="Dense connections between layers, improved gradient flow",
        parameter_count=7_978_856,
        typical_training_time_per_epoch_seconds=45,
    ),
    NetworkArchitecture(
        id=NetworkType.DENSENET169,
        name="DenseNet-169",
        description="Deeper DenseNet with 169 layers",
        parameter_count=14_149_480,
        typical_training_time_per_epoch_seconds=60,
    ),
    NetworkArchitecture(
        id=NetworkType.DENSENET201,
        name="DenseNet-201",
        description="Deepest standard DenseNet with 201 layers",
        parameter_count=20_013_928,
        typical_training_time_per_epoch_seconds=75,
    ),
    NetworkArchitecture(
        id=NetworkType.MOBILENET_V2,
        name="MobileNet V2",
        description="Efficient mobile architecture with inverted residuals",
        parameter_count=3_504_872,
        typical_training_time_per_epoch_seconds=25,
    ),
    NetworkArchitecture(
        id=NetworkType.MOBILENET_V3_SMALL,
        name="MobileNet V3 Small",
        description="Small MobileNet V3 variant, optimized for mobile devices",
        parameter_count=2_542_856,
        typical_training_time_per_epoch_seconds=20,
    ),
    NetworkArchitecture(
        id=NetworkType.MOBILENET_V3_LARGE,
        name="MobileNet V3 Large",
        description="Large MobileNet V3 variant with better accuracy",
        parameter_count=5_483_032,
        typical_training_time_per_epoch_seconds=30,
    ),
    NetworkArchitecture(
        id=NetworkType.CONVNEXT_TINY,
        name="ConvNeXt-Tiny",
        description="Modern CNN architecture competitive with transformers",
        parameter_count=28_589_128,
        typical_training_time_per_epoch_seconds=55,
    ),
    NetworkArchitecture(
        id=NetworkType.CONVNEXT_SMALL,
        name="ConvNeXt-Small",
        description="Larger ConvNeXt variant with improved accuracy",
        parameter_count=50_223_688,
        typical_training_time_per_epoch_seconds=80,
    ),
    NetworkArchitecture(
        id=NetworkType.INCEPTION_V3,
        name="Inception V3",
        description="Multi-scale convolutions with factorized filters",
        parameter_count=27_161_264,
        typical_training_time_per_epoch_seconds=65,
    ),
    NetworkArchitecture(
        id=NetworkType.SWIN_TINY,
        name="Swin Transformer Tiny",
        description="Hierarchical vision transformer with shifted windows",
        parameter_count=28_288_354,
        typical_training_time_per_epoch_seconds=70,
    ),
    NetworkArchitecture(
        id=NetworkType.SWIN_SMALL,
        name="Swin Transformer Small",
        description="Larger Swin Transformer with better performance",
        parameter_count=49_606_258,
        typical_training_time_per_epoch_seconds=95,
    ),
    NetworkArchitecture(
        id=NetworkType.MAXVIT_T,
        name="MaxViT-Tiny",
        description="Hybrid architecture combining convolution and attention",
        parameter_count=30_919_624,
        typical_training_time_per_epoch_seconds=85,
    ),
    NetworkArchitecture(
        id=NetworkType.UNET,
        name="U-Net",
        description="Encoder-decoder architecture with skip connections (adapted for classification)",
        parameter_count=31_042_442,
        typical_training_time_per_epoch_seconds=60,
    ),
]

# Available initialization methods
INITIALIZATION_OPTIONS: list[InitializationOption] = [
    InitializationOption(
        id=InitializationMethod.RANDOM_NORMAL,
        name="Random Normal",
        description="Random values from normal distribution (mean=0, std=0.01)",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.RANDOM_UNIFORM,
        name="Random Uniform",
        description="Random values uniformly distributed between -0.05 and 0.05",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.XAVIER_NORMAL,
        name="Xavier Normal (Glorot)",
        description="Normal distribution scaled by layer fan-in/fan-out",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.XAVIER_UNIFORM,
        name="Xavier Uniform (Glorot)",
        description="Uniform distribution scaled by layer fan-in/fan-out",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.KAIMING_NORMAL,
        name="Kaiming Normal (He)",
        description="Normal distribution for ReLU networks (fan-in scaling)",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.KAIMING_UNIFORM,
        name="Kaiming Uniform (He)",
        description="Uniform distribution for ReLU networks (fan-in scaling)",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.PRETRAINED_IMAGENET,
        name="Pretrained ImageNet",
        description="Weights pretrained on ImageNet dataset",
        requires_pretrained=True,
    ),
    InitializationOption(
        id=InitializationMethod.ZEROS,
        name="Zeros",
        description="Initialize all weights to zero (not recommended for hidden layers)",
        requires_pretrained=False,
    ),
    InitializationOption(
        id=InitializationMethod.ONES,
        name="Ones",
        description="Initialize all weights to one (not recommended)",
        requires_pretrained=False,
    ),
]


# Default optimizer configurations
DEFAULT_OPTIMIZER_CONFIGS = {
    OptimizerType.SGD: OptimizerConfig(
        type=OptimizerType.SGD,
        learning_rate=0.01,
        weight_decay=0.0001,
        momentum=0.9,
        nesterov=True,
    ),
    OptimizerType.ADAM: OptimizerConfig(
        type=OptimizerType.ADAM,
        learning_rate=0.001,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ),
    OptimizerType.ADAMW: OptimizerConfig(
        type=OptimizerType.ADAMW,
        learning_rate=0.001,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
    ),
}
