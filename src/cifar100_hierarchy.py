"""
CIFAR-100 superclass hierarchy mapping and utilities.

Provides the canonical CIFAR-100 20-superclass → 100-fine-class mapping
and helper functions for building index-based hierarchy maps.
"""
from typing import Dict, List


# Official CIFAR-100 superclass → fine-class mapping
# Reference: https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR100_SUPERCLASSES: Dict[str, List[str]] = {
    "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
    "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
    "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
    "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
    "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
    "household_electrical_devices": ["clock", "keyboard", "lamp", "telephone", "television"],
    "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
    "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
    "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
    "large_man-made_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
    "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
    "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
    "medium_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
    "non-insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
    "people": ["baby", "boy", "girl", "man", "woman"],
    "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
    "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
    "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
    "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
    "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
}

# Reverse mapping: fine class name → superclass name
_FINE_TO_SUPER: Dict[str, str] = {}
for _super, _fines in CIFAR100_SUPERCLASSES.items():
    for _fine in _fines:
        _FINE_TO_SUPER[_fine] = _super


def build_superclass_map(
    selected_classes: List[str],
    class_to_idx: Dict[str, int],
) -> Dict[int, str]:
    """
    Map each class index to its CIFAR-100 superclass name.

    Args:
        selected_classes: List of fine-class names used in the experiment.
        class_to_idx: Mapping from class name to integer index.

    Returns:
        Dict mapping class_idx → superclass name.
        Classes not found in CIFAR-100 hierarchy are mapped to "unknown".
    """
    superclass_map: Dict[int, str] = {}
    for class_name in selected_classes:
        idx = class_to_idx.get(class_name)
        if idx is None:
            continue
        # Strip dataset prefix (e.g. "cifar100_beaver" → "beaver")
        bare_name = class_name.split("_", 1)[1] if "_" in class_name else class_name
        superclass_map[idx] = _FINE_TO_SUPER.get(bare_name, _FINE_TO_SUPER.get(class_name, "unknown"))
    return superclass_map


def get_superclass_groups(
    superclass_map: Dict[int, str],
) -> Dict[str, List[int]]:
    """
    Group class indices by their superclass.

    Args:
        superclass_map: Dict mapping class_idx → superclass name
            (as returned by build_superclass_map).

    Returns:
        Dict mapping superclass name → list of class indices in that superclass.
    """
    groups: Dict[str, List[int]] = {}
    for class_idx, superclass_name in superclass_map.items():
        groups.setdefault(superclass_name, []).append(class_idx)
    # Sort indices within each group for determinism
    for key in groups:
        groups[key].sort()
    return groups


def has_hierarchical_structure(
    superclass_map: Dict[int, str],
) -> bool:
    """
    Check whether the selected classes have meaningful hierarchical structure.

    Returns True if at least 2 classes share a common superclass,
    which is the minimum needed for within-superclass analyses.
    """
    groups = get_superclass_groups(superclass_map)
    return any(len(indices) >= 2 for indices in groups.values())
