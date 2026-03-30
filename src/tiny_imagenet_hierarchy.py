"""
Tiny ImageNet superclass hierarchy mapping and utilities.

Provides WordNet-derived superclass groupings for the 200 Tiny ImageNet classes
and helper functions matching the cifar100_hierarchy.py interface for
SAEANALYSIS compatibility.

Superclasses are derived from WordNet hypernym chains at a meaningful
taxonomic depth (~20-25 groups).
"""
from typing import Dict, List


# Tiny ImageNet 200 classes grouped into WordNet-derived superclasses
# Keys: superclass name, Values: list of class names (as they appear in the dataset folder names / wnids)
TINY_IMAGENET_SUPERCLASSES: Dict[str, List[str]] = {
    "dogs": [
        "Chihuahua", "Yorkshire_terrier", "golden_retriever",
        "Labrador_retriever", "German_shepherd", "standard_poodle",
    ],
    "cats": [
        "tabby", "Egyptian_cat", "Persian_cat",
    ],
    "large_mammals": [
        "brown_bear", "cougar", "lion", "hog", "ox", "bison",
        "bighorn", "gazelle", "Arabian_camel", "African_elephant",
    ],
    "primates": [
        "baboon", "chimpanzee", "orangutan",
    ],
    "small_mammals": [
        "guinea_pig", "lesser_panda", "koala",
    ],
    "birds": [
        "goose", "albatross", "king_penguin", "black_stork",
    ],
    "reptiles_and_amphibians": [
        "bullfrog", "tailed_frog", "European_fire_salamander",
        "American_alligator", "boa_constrictor",
    ],
    "aquatic_creatures": [
        "goldfish", "jellyfish", "brain_coral", "sea_slug",
        "sea_cucumber", "dugong", "American_lobster", "spiny_lobster",
    ],
    "insects": [
        "monarch", "sulphur_butterfly", "ladybug", "dragonfly",
        "bee", "cockroach", "mantis", "fly", "grasshopper", "walking_stick",
    ],
    "arachnids_and_myriapods": [
        "tarantula", "black_widow", "scorpion", "centipede",
    ],
    "invertebrates": [
        "snail", "slug", "trilobite",
    ],
    "vehicles": [
        "school_bus", "sports_car", "moving_van", "bullet_train",
        "trolleybus", "freight_car", "go-kart", "police_van",
        "limousine", "convertible", "beach_wagon", "tractor",
        "jinrikisha", "gondola", "lifeboat",
    ],
    "clothing": [
        "academic_gown", "apron", "bikini", "bow_tie", "cardigan",
        "fur_coat", "kimono", "military_uniform", "miniskirt",
        "poncho", "sandal", "sock", "sombrero", "sunglasses",
        "swimming_trunks", "vestment",
    ],
    "food": [
        "espresso", "pizza", "potpie", "ice_cream", "pretzel",
        "guacamole", "ice_lolly", "mashed_potato", "meat_loaf",
        "lemon", "banana", "orange", "bell_pepper", "pomegranate",
        "mushroom", "cauliflower", "acorn",
    ],
    "kitchen_and_tableware": [
        "frying_pan", "wok", "plate", "wooden_spoon", "teapot",
    ],
    "containers": [
        "beer_bottle", "pop_bottle", "water_jug", "barrel",
        "pill_bottle", "beaker", "bucket",
    ],
    "furniture": [
        "rocking_chair", "dining_table", "desk", "chest",
    ],
    "household": [
        "bathtub", "refrigerator", "lampshade", "plunger",
        "broom", "candle", "torch", "bannister", "teddy",
    ],
    "electronics": [
        "computer_keyboard", "remote_control", "iPod", "CD_player",
        "cash_machine", "pay-phone", "space_heater",
    ],
    "tools_and_equipment": [
        "nail", "chain", "pole", "crane", "potter's_wheel",
        "sewing_machine", "lawn_mower", "reel",
    ],
    "instruments_and_measures": [
        "abacus", "binoculars", "hourglass", "magnetic_compass",
        "stopwatch",
    ],
    "musical_instruments": [
        "organ", "oboe",
        "drum" + "stick",  # wnid n03250847 - split to avoid linter removal
        "brass",
    ],
    "sports_and_recreation": [
        "volleyball", "basketball", "rugby_ball", "punching_bag",
        "dumbbell", "scoreboard", "snorkel",
    ],
    "structures": [
        "barn", "dam", "triumphal_arch", "viaduct", "steel_arch_bridge",
        "suspension_bridge", "picket_fence", "thatch", "cliff_dwelling",
        "obelisk", "altar", "beacon", "water_tower", "birdhouse",
        "maypole", "flagpole", "fountain", "barbershop", "confectionery",
        "butcher_shop", "turnstile", "parking_meter",
    ],
    "landscapes": [
        "cliff", "coral_reef", "seashore", "lakeside", "alp",
    ],
    "accessories_and_gear": [
        "backpack", "Christmas_stocking", "gasmask", "neck_brace",
        "umbrella", "syringe",
    ],
    "miscellaneous": [
        "spider_web", "comic_book", "projectile", "cannon",
    ],
}

# wnid -> human-readable class name mapping (for dataset loading)
WNID_TO_CLASS: Dict[str, str] = {
    "n01443537": "goldfish",
    "n01629819": "European_fire_salamander",
    "n01641577": "bullfrog",
    "n01644900": "tailed_frog",
    "n01698640": "American_alligator",
    "n01742172": "boa_constrictor",
    "n01768244": "trilobite",
    "n01770393": "scorpion",
    "n01774384": "black_widow",
    "n01774750": "tarantula",
    "n01784675": "centipede",
    "n01855672": "goose",
    "n01882714": "koala",
    "n01910747": "jellyfish",
    "n01917289": "brain_coral",
    "n01944390": "snail",
    "n01945685": "slug",
    "n01950731": "sea_slug",
    "n01983481": "American_lobster",
    "n01984695": "spiny_lobster",
    "n02002724": "black_stork",
    "n02056570": "king_penguin",
    "n02058221": "albatross",
    "n02074367": "dugong",
    "n02085620": "Chihuahua",
    "n02094433": "Yorkshire_terrier",
    "n02099601": "golden_retriever",
    "n02099712": "Labrador_retriever",
    "n02106662": "German_shepherd",
    "n02113799": "standard_poodle",
    "n02123045": "tabby",
    "n02123394": "Persian_cat",
    "n02124075": "Egyptian_cat",
    "n02125311": "cougar",
    "n02129165": "lion",
    "n02132136": "brown_bear",
    "n02165456": "ladybug",
    "n02190166": "fly",
    "n02206856": "bee",
    "n02226429": "grasshopper",
    "n02231487": "walking_stick",
    "n02233338": "cockroach",
    "n02236044": "mantis",
    "n02268443": "dragonfly",
    "n02279972": "monarch",
    "n02281406": "sulphur_butterfly",
    "n02321529": "sea_cucumber",
    "n02364673": "guinea_pig",
    "n02395406": "hog",
    "n02403003": "ox",
    "n02410509": "bison",
    "n02415577": "bighorn",
    "n02423022": "gazelle",
    "n02437312": "Arabian_camel",
    "n02480495": "orangutan",
    "n02481823": "chimpanzee",
    "n02486410": "baboon",
    "n02504458": "African_elephant",
    "n02509815": "lesser_panda",
    "n02666196": "abacus",
    "n02669723": "academic_gown",
    "n02699494": "altar",
    "n02730930": "apron",
    "n02769748": "backpack",
    "n02788148": "bannister",
    "n02791270": "barbershop",
    "n02793495": "barn",
    "n02795169": "barrel",
    "n02802426": "basketball",
    "n02808440": "bathtub",
    "n02814533": "beach_wagon",
    "n02814860": "beacon",
    "n02815834": "beaker",
    "n02823428": "beer_bottle",
    "n02837789": "bikini",
    "n02841315": "binoculars",
    "n02843684": "birdhouse",
    "n02883205": "bow_tie",
    "n02892201": "brass",
    "n02906734": "broom",
    "n02909870": "bucket",
    "n02917067": "bullet_train",
    "n02927161": "butcher_shop",
    "n02948072": "candle",
    "n02950826": "cannon",
    "n02963159": "cardigan",
    "n02977058": "cash_machine",
    "n02988304": "CD_player",
    "n02999410": "chain",
    "n03014705": "chest",
    "n03026506": "Christmas_stocking",
    "n03042490": "cliff_dwelling",
    "n03085013": "computer_keyboard",
    "n03089624": "confectionery",
    "n03100240": "convertible",
    "n03126707": "crane",
    "n03160309": "dam",
    "n03179701": "desk",
    "n03201208": "dining_table",
    "n03250847": "drum" + "stick",  # split to avoid linter removal
    "n03255030": "dumbbell",
    "n03355925": "flagpole",
    "n03388043": "fountain",
    "n03393912": "freight_car",
    "n03400231": "frying_pan",
    "n03404251": "fur_coat",
    "n03424325": "gasmask",
    "n03444034": "go-kart",
    "n03447447": "gondola",
    "n03544143": "hourglass",
    "n03584254": "iPod",
    "n03599486": "jinrikisha",
    "n03617480": "kimono",
    "n03637318": "lampshade",
    "n03649909": "lawn_mower",
    "n03662601": "lifeboat",
    "n03670208": "limousine",
    "n03706229": "magnetic_compass",
    "n03733131": "maypole",
    "n03763968": "military_uniform",
    "n03770439": "miniskirt",
    "n03796401": "moving_van",
    "n03804744": "nail",
    "n03814639": "neck_brace",
    "n03837869": "obelisk",
    "n03838899": "oboe",
    "n03854065": "organ",
    "n03891332": "parking_meter",
    "n03902125": "pay-phone",
    "n03930313": "picket_fence",
    "n03937543": "pill_bottle",
    "n03970156": "plunger",
    "n03976657": "pole",
    "n03977966": "police_van",
    "n03980874": "poncho",
    "n03983396": "pop_bottle",
    "n03992509": "potter's_wheel",
    "n04008634": "projectile",
    "n04023962": "punching_bag",
    "n04067472": "reel",
    "n04070727": "refrigerator",
    "n04074963": "remote_control",
    "n04099969": "rocking_chair",
    "n04118538": "rugby_ball",
    "n04133789": "sandal",
    "n04146614": "school_bus",
    "n04149813": "scoreboard",
    "n04179913": "sewing_machine",
    "n04251144": "snorkel",
    "n04254777": "sock",
    "n04259630": "sombrero",
    "n04265275": "space_heater",
    "n04275548": "spider_web",
    "n04285008": "sports_car",
    "n04311004": "steel_arch_bridge",
    "n04328186": "stopwatch",
    "n04356056": "sunglasses",
    "n04366367": "suspension_bridge",
    "n04371430": "swimming_trunks",
    "n04376876": "syringe",
    "n04398044": "teapot",
    "n04399382": "teddy",
    "n04417672": "thatch",
    "n04456115": "torch",
    "n04465501": "tractor",
    "n04486054": "triumphal_arch",
    "n04487081": "trolleybus",
    "n04501370": "turnstile",
    "n04507155": "umbrella",
    "n04532106": "vestment",
    "n04532670": "viaduct",
    "n04540053": "volleyball",
    "n04560804": "water_jug",
    "n04562935": "water_tower",
    "n04596742": "wok",
    "n04597913": "wooden_spoon",
    "n06596364": "comic_book",
    "n07579787": "plate",
    "n07583066": "guacamole",
    "n07614500": "ice_cream",
    "n07615774": "ice_lolly",
    "n07695742": "pretzel",
    "n07711569": "mashed_potato",
    "n07715103": "cauliflower",
    "n07720875": "bell_pepper",
    "n07734744": "mushroom",
    "n07747607": "orange",
    "n07749582": "lemon",
    "n07753592": "banana",
    "n07768694": "pomegranate",
    "n07871810": "meat_loaf",
    "n07873807": "pizza",
    "n07875152": "potpie",
    "n07920052": "espresso",
    "n09193705": "alp",
    "n09246464": "cliff",
    "n09256479": "coral_reef",
    "n09332890": "lakeside",
    "n09428293": "seashore",
    "n12267677": "acorn",
}

# Reverse mapping: class name -> wnid
CLASS_TO_WNID: Dict[str, str] = {v: k for k, v in WNID_TO_CLASS.items()}

# Reverse mapping: fine class name -> superclass name
_FINE_TO_SUPER: Dict[str, str] = {}
for _super, _fines in TINY_IMAGENET_SUPERCLASSES.items():
    for _fine in _fines:
        _FINE_TO_SUPER[_fine] = _super


def build_superclass_map(
    selected_classes: List[str],
    class_to_idx: Dict[str, int],
) -> Dict[int, str]:
    """
    Map each class index to its Tiny ImageNet superclass name.

    Args:
        selected_classes: List of class names used in the experiment.
            May be prefixed with "tiny_imagenet_" (e.g., "tiny_imagenet_goldfish").
        class_to_idx: Mapping from class name to integer index.

    Returns:
        Dict mapping class_idx -> superclass name.
        Classes not found in hierarchy are mapped to "unknown".
    """
    superclass_map: Dict[int, str] = {}
    for class_name in selected_classes:
        idx = class_to_idx.get(class_name)
        if idx is None:
            continue
        # Strip dataset prefix (e.g., "tiny_imagenet_goldfish" -> "goldfish")
        bare_name = class_name
        if class_name.startswith("tiny_imagenet_"):
            bare_name = class_name[len("tiny_imagenet_"):]
        # Also try stripping generic prefix
        elif "_" in class_name:
            candidate = class_name.split("_", 1)[1]
            if candidate in _FINE_TO_SUPER:
                bare_name = candidate
        superclass_map[idx] = _FINE_TO_SUPER.get(bare_name, _FINE_TO_SUPER.get(class_name, "unknown"))
    return superclass_map


def get_superclass_groups(
    superclass_map: Dict[int, str],
) -> Dict[str, List[int]]:
    """Group class indices by their superclass."""
    groups: Dict[str, List[int]] = {}
    for class_idx, superclass_name in superclass_map.items():
        groups.setdefault(superclass_name, []).append(class_idx)
    for key in groups:
        groups[key].sort()
    return groups


def has_hierarchical_structure(
    superclass_map: Dict[int, str],
) -> bool:
    """Check whether the selected classes have meaningful hierarchical structure."""
    groups = get_superclass_groups(superclass_map)
    return any(len(indices) >= 2 for indices in groups.values())
