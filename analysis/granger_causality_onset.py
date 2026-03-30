#!/usr/bin/env python3
"""
Panel Granger Causality: Per-Superclass Onset Timing

Tests whether superclasses whose Ab-E forms earlier also show earlier Di-E onset,
and whether Tg-E onset precedes Ab-E/Di-E onset — across CIFAR-100 (9 lanes × 20 SC)
and Tiny ImageNet (1 lane × 28 SC).

Panel structure:
  Cross-sectional unit: (dataset, lane, superclass)
  Time dimension: transitions (t = 0→1, 1→2, ..., 8→terminal)

Analyses:
  1. Panel Granger: Ab-E(t) → Di-E(t+1), Tg-E(t) → Ab-E(t+1), Tg-E(t) → Di-E(t+1)
     with dataset + superclass + lane fixed effects
  2. Onset timing: per-superclass peak/onset transition, cross-sectional correlation
  3. Lag consistency: Ab-E peak → Di-E peak lag distribution
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, f as f_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────
EXT_BASE = os.environ.get("RDT_DATA_ROOT", "/Volumes/ExternalDrive/EmpiricalSignatures_data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "output", "figures")

# ── CIFAR-100 superclass mapping ──────────────────────────────────────────
CIFAR100_COARSE_TO_FINE = {
    0: [4, 30, 55, 72, 95],     1: [1, 32, 67, 73, 91],
    2: [54, 62, 70, 82, 92],    3: [9, 10, 16, 28, 61],
    4: [0, 51, 53, 57, 83],     5: [22, 39, 40, 86, 87],
    6: [5, 20, 25, 84, 94],     7: [6, 7, 14, 18, 24],
    8: [3, 42, 43, 88, 97],     9: [12, 17, 37, 68, 76],
    10: [23, 33, 49, 60, 71],   11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75],   13: [26, 45, 77, 79, 99],
    14: [2, 11, 35, 46, 98],    15: [27, 29, 44, 78, 93],
    16: [36, 50, 65, 74, 80],   17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90],    19: [41, 69, 81, 85, 89],
}
CIFAR100_SC_NAMES = [
    'aquatic_mammals', 'fish', 'flowers', 'food_containers', 'fruit_and_vegetables',
    'household_electrical_devices', 'household_furniture', 'insects', 'large_carnivores',
    'large_man-made_outdoor_things', 'large_natural_outdoor_scenes', 'large_omnivores_and_herbivores',
    'medium_mammals', 'non-insect_invertebrates', 'people', 'reptiles', 'small_mammals',
    'trees', 'vehicles_1', 'vehicles_2'
]
CIFAR100_FINE_TO_COARSE = {}
for ci, fines in CIFAR100_COARSE_TO_FINE.items():
    for fi in fines:
        CIFAR100_FINE_TO_COARSE[fi] = ci

# ── Tiny ImageNet superclass mapping ─────────────────────────────────────
# Build class_idx → superclass mapping from wnid ordering
TINY_SUPERCLASSES = {
    "dogs": ["Chihuahua", "Yorkshire_terrier", "golden_retriever",
             "Labrador_retriever", "German_shepherd", "standard_poodle"],
    "cats": ["tabby", "Egyptian_cat", "Persian_cat"],
    "large_mammals": ["brown_bear", "cougar", "lion", "hog", "ox", "bison",
                      "bighorn", "gazelle", "Arabian_camel", "African_elephant"],
    "primates": ["baboon", "chimpanzee", "orangutan"],
    "small_mammals": ["guinea_pig", "lesser_panda", "koala"],
    "birds": ["goose", "albatross", "king_penguin", "black_stork"],
    "reptiles_and_amphibians": ["bullfrog", "tailed_frog", "European_fire_salamander",
                                "American_alligator", "boa_constrictor"],
    "aquatic_creatures": ["goldfish", "jellyfish", "brain_coral", "sea_slug",
                          "sea_cucumber", "dugong", "American_lobster", "spiny_lobster"],
    "insects": ["monarch", "sulphur_butterfly", "ladybug", "dragonfly",
                "bee", "cockroach", "mantis", "fly", "grasshopper", "walking_stick"],
    "arachnids_and_myriapods": ["tarantula", "black_widow", "scorpion", "centipede"],
    "invertebrates": ["snail", "slug", "trilobite"],
    "vehicles": ["school_bus", "sports_car", "moving_van", "bullet_train",
                 "trolleybus", "freight_car", "go-kart", "police_van",
                 "limousine", "convertible", "beach_wagon", "tractor",
                 "jinrikisha", "gondola", "lifeboat"],
    "clothing": ["academic_gown", "apron", "bikini", "bow_tie", "cardigan",
                 "fur_coat", "kimono", "military_uniform", "miniskirt",
                 "poncho", "sandal", "sock", "sombrero", "sunglasses",
                 "swimming_trunks", "vestment"],
    "food": ["espresso", "pizza", "potpie", "ice_cream", "pretzel",
             "guacamole", "ice_lolly", "mashed_potato", "meat_loaf",
             "lemon", "banana", "orange", "bell_pepper", "pomegranate",
             "mushroom", "cauliflower", "acorn"],
    "kitchen_and_tableware": ["frying_pan", "wok", "plate", "wooden_spoon", "teapot"],
    "containers": ["beer_bottle", "pop_bottle", "water_jug", "barrel",
                   "pill_bottle", "beaker", "bucket"],
    "furniture": ["rocking_chair", "dining_table", "desk", "chest"],
    "household": ["bathtub", "refrigerator", "lampshade", "plunger",
                  "broom", "candle", "torch", "bannister", "teddy"],
    "electronics": ["computer_keyboard", "remote_control", "iPod", "CD_player",
                    "cash_machine", "pay-phone", "space_heater"],
    "tools_and_equipment": ["nail", "chain", "pole", "crane", "potter's_wheel",
                            "sewing_machine", "lawn_mower", "reel"],
    "instruments_and_measures": ["abacus", "binoculars", "hourglass", "magnetic_compass",
                                 "stopwatch"],
    "musical_instruments": ["organ", "oboe", "drumstick", "brass"],
    "sports_and_recreation": ["volleyball", "basketball", "rugby_ball", "punching_bag",
                              "dumbbell", "scoreboard", "snorkel"],
    "structures": ["barn", "dam", "triumphal_arch", "viaduct", "steel_arch_bridge",
                   "suspension_bridge", "picket_fence", "thatch", "cliff_dwelling",
                   "obelisk", "altar", "beacon", "water_tower", "birdhouse",
                   "maypole", "flagpole", "fountain", "barbershop", "confectionery",
                   "butcher_shop", "turnstile", "parking_meter"],
    "landscapes": ["cliff", "coral_reef", "seashore", "lakeside", "alp"],
    "accessories_and_gear": ["backpack", "Christmas_stocking", "gasmask", "neck_brace",
                             "umbrella", "syringe"],
    "miscellaneous": ["spider_web", "comic_book", "projectile", "cannon"],
}

WNID_TO_CLASS = {
    "n01443537": "goldfish", "n01629819": "European_fire_salamander",
    "n01641577": "bullfrog", "n01644900": "tailed_frog",
    "n01698640": "American_alligator", "n01742172": "boa_constrictor",
    "n01768244": "trilobite", "n01770393": "scorpion",
    "n01774384": "black_widow", "n01774750": "tarantula",
    "n01784675": "centipede", "n01855672": "goose",
    "n01882714": "koala", "n01910747": "jellyfish",
    "n01917289": "brain_coral", "n01944390": "snail",
    "n01945685": "slug", "n01950731": "sea_slug",
    "n01983481": "American_lobster", "n01984695": "spiny_lobster",
    "n02002724": "black_stork", "n02056570": "king_penguin",
    "n02058221": "albatross", "n02074367": "dugong",
    "n02085620": "Chihuahua", "n02094433": "Yorkshire_terrier",
    "n02099601": "golden_retriever", "n02099712": "Labrador_retriever",
    "n02106662": "German_shepherd", "n02113799": "standard_poodle",
    "n02123045": "tabby", "n02123394": "Persian_cat",
    "n02124075": "Egyptian_cat", "n02125311": "cougar",
    "n02129165": "lion", "n02132136": "brown_bear",
    "n02165456": "ladybug", "n02190166": "fly",
    "n02206856": "bee", "n02226429": "grasshopper",
    "n02231487": "walking_stick", "n02233338": "cockroach",
    "n02236044": "mantis", "n02268443": "dragonfly",
    "n02279972": "monarch", "n02281406": "sulphur_butterfly",
    "n02321529": "sea_cucumber", "n02364673": "guinea_pig",
    "n02395406": "hog", "n02403003": "ox",
    "n02410509": "bison", "n02415577": "bighorn",
    "n02423022": "gazelle", "n02437312": "Arabian_camel",
    "n02480495": "orangutan", "n02481823": "chimpanzee",
    "n02486410": "baboon", "n02504458": "African_elephant",
    "n02509815": "lesser_panda", "n02666196": "abacus",
    "n02669723": "academic_gown", "n02699494": "altar",
    "n02730930": "apron", "n02769748": "backpack",
    "n02788148": "bannister", "n02791270": "barbershop",
    "n02793495": "barn", "n02795169": "barrel",
    "n02802426": "basketball", "n02808440": "bathtub",
    "n02814533": "beach_wagon", "n02814860": "beacon",
    "n02815834": "beaker", "n02823428": "beer_bottle",
    "n02837789": "bikini", "n02841315": "binoculars",
    "n02843684": "birdhouse", "n02883205": "bow_tie",
    "n02892201": "brass", "n02906734": "broom",
    "n02909870": "bucket", "n02917067": "bullet_train",
    "n02927161": "butcher_shop", "n02948072": "candle",
    "n02950826": "cannon", "n02963159": "cardigan",
    "n02977058": "cash_machine", "n02988304": "CD_player",
    "n02999410": "chain", "n03014705": "chest",
    "n03026506": "Christmas_stocking", "n03042490": "cliff_dwelling",
    "n03085013": "computer_keyboard", "n03089624": "confectionery",
    "n03100240": "convertible", "n03126707": "crane",
    "n03160309": "dam", "n03179701": "desk",
    "n03201208": "dining_table", "n03250847": "drumstick",
    "n03255030": "dumbbell", "n03355925": "flagpole",
    "n03388043": "fountain", "n03393912": "freight_car",
    "n03400231": "frying_pan", "n03404251": "fur_coat",
    "n03424325": "gasmask", "n03444034": "go-kart",
    "n03447447": "gondola", "n03544143": "hourglass",
    "n03584254": "iPod", "n03599486": "jinrikisha",
    "n03617480": "kimono", "n03637318": "lampshade",
    "n03649909": "lawn_mower", "n03662601": "lifeboat",
    "n03670208": "limousine", "n03706229": "magnetic_compass",
    "n03733131": "maypole", "n03763968": "military_uniform",
    "n03770439": "miniskirt", "n03796401": "moving_van",
    "n03804744": "nail", "n03814639": "neck_brace",
    "n03837869": "obelisk", "n03838899": "oboe",
    "n03854065": "organ", "n03891332": "parking_meter",
    "n03902125": "pay-phone", "n03930313": "picket_fence",
    "n03937543": "pill_bottle", "n03970156": "plunger",
    "n03976657": "pole", "n03977966": "police_van",
    "n03980874": "poncho", "n03983396": "pop_bottle",
    "n03992509": "potter's_wheel", "n04008634": "projectile",
    "n04023962": "punching_bag", "n04067472": "reel",
    "n04070727": "refrigerator", "n04074963": "remote_control",
    "n04099969": "rocking_chair", "n04118538": "rugby_ball",
    "n04133789": "sandal", "n04146614": "school_bus",
    "n04149813": "scoreboard", "n04179913": "sewing_machine",
    "n04251144": "snorkel", "n04254777": "sock",
    "n04259630": "sombrero", "n04265275": "space_heater",
    "n04275548": "spider_web", "n04285008": "sports_car",
    "n04311004": "steel_arch_bridge", "n04328186": "stopwatch",
    "n04356056": "sunglasses", "n04366367": "suspension_bridge",
    "n04371430": "swimming_trunks", "n04376876": "syringe",
    "n04398044": "teapot", "n04399382": "teddy",
    "n04417672": "thatch", "n04456115": "torch",
    "n04465501": "tractor", "n04486054": "triumphal_arch",
    "n04487081": "trolleybus", "n04501370": "turnstile",
    "n04507155": "umbrella", "n04532106": "vestment",
    "n04532670": "viaduct", "n04540053": "volleyball",
    "n04560804": "water_jug", "n04562935": "water_tower",
    "n04596742": "wok", "n04597913": "wooden_spoon",
    "n06596364": "comic_book", "n07579787": "plate",
    "n07583066": "guacamole", "n07614500": "ice_cream",
    "n07615774": "ice_lolly", "n07695742": "pretzel",
    "n07711569": "mashed_potato", "n07715103": "cauliflower",
    "n07720875": "bell_pepper", "n07734744": "mushroom",
    "n07747607": "orange", "n07749582": "lemon",
    "n07753592": "banana", "n07768694": "pomegranate",
    "n07871810": "meat_loaf", "n07873807": "pizza",
    "n07875152": "potpie", "n07920052": "espresso",
    "n09193705": "alp", "n09246464": "cliff",
    "n09256479": "coral_reef", "n09332890": "lakeside",
    "n09428293": "seashore", "n12267677": "acorn",
}

# Build Tiny ImageNet fine→superclass from sorted wnids (ImageFolder ordering)
TINY_SORTED_WNIDS = sorted(WNID_TO_CLASS.keys())
TINY_FINE_TO_SUPER = {}  # class_name → superclass_name
for sc_name, class_names in TINY_SUPERCLASSES.items():
    for cn in class_names:
        TINY_FINE_TO_SUPER[cn] = sc_name

TINY_IDX_TO_SUPER = {}  # class_idx → superclass_name
for idx, wnid in enumerate(TINY_SORTED_WNIDS):
    class_name = WNID_TO_CLASS[wnid]
    TINY_IDX_TO_SUPER[idx] = TINY_FINE_TO_SUPER.get(class_name, "unknown")

TINY_SC_NAMES = sorted(set(TINY_IDX_TO_SUPER.values()) - {"unknown"})

# Tiny ImageNet lane info
TINY_EXP_ID = "821eaadb-0160-48f8-9669-ec3d9185c428"
TINY_LANE_ID = "45476b66-90e0-47f0-aec6-41c62aff8c21"
TINY_LANE_NAME = "ResNet18-TinyImageNet-200class-seed42"


def extract_transition_data(lane_id, exp_id, lane_name, fine_to_super_fn, sc_names):
    """Extract per-superclass per-transition process counts from raw transition files.

    Args:
        fine_to_super_fn: callable(class_idx) → superclass_name or None
        sc_names: list of superclass names
    """
    trans_dir = os.path.join(EXT_BASE, "experiments", exp_id, "lanes", lane_id,
                             "sae_analysis", "transitions")
    if not os.path.exists(trans_dir):
        print(f"  SKIP {lane_name}: no transitions dir at {trans_dir}")
        return None

    transition_names = sorted(os.listdir(trans_dir))
    def sort_key(name):
        parts = name.split("_to_")
        try:
            return int(parts[0])
        except ValueError:
            return 999
    transition_names = sorted([t for t in transition_names
                                if os.path.isdir(os.path.join(trans_dir, t))],
                               key=sort_key)

    results = []
    for trans_name in transition_names:
        trans_path = os.path.join(trans_dir, trans_name)
        parts = trans_name.split("_to_")
        trans_idx = int(parts[0])

        sc_counts = defaultdict(lambda: {"ab_h": 0, "di_h": 0, "tg_h": 0,
                                          "as_h": 0, "de_h": 0, "total": 0})

        layer_files = [f for f in os.listdir(trans_path) if f.endswith(".json")]
        for layer_file in layer_files:
            with open(os.path.join(trans_path, layer_file)) as f:
                samples = json.load(f)
            for sample in samples:
                class_idx = sample["class_idx"]
                sc_name = fine_to_super_fn(class_idx)
                if sc_name is None:
                    continue
                pc = sample["process_counts"]
                sc_counts[sc_name]["ab_h"] += pc.get("ab_h", 0)
                sc_counts[sc_name]["di_h"] += pc.get("di_h", 0)
                sc_counts[sc_name]["tg_h"] += pc.get("tg_h", 0)
                sc_counts[sc_name]["as_h"] += pc.get("as_h", 0)
                sc_counts[sc_name]["de_h"] += pc.get("de_h", 0)
                sc_counts[sc_name]["total"] += sum(pc.get(k, 0) for k in
                                                    ["ab_h", "di_h", "tg_h", "as_h", "de_h"])

        for sc_name in sc_names:
            counts = sc_counts[sc_name]
            results.append({
                "lane": lane_name,
                "transition": trans_idx,
                "superclass": sc_name,
                "ab_h": counts["ab_h"],
                "di_h": counts["di_h"],
                "tg_h": counts["tg_h"],
                "as_h": counts["as_h"],
                "de_h": counts["de_h"],
                "total": counts["total"],
            })

    return results


def load_cifar100_data():
    """Load CIFAR-100 data from cache, mapping superclass_idx → superclass name."""
    cached_path = os.path.join(OUTPUT_DIR, "superclass_transition_series.json")
    if not os.path.exists(cached_path):
        print("ERROR: CIFAR-100 cache not found. Run granger_causality_superclass.py first.")
        sys.exit(1)
    df = pd.read_json(cached_path)
    # Ensure superclass column is the name string
    if "superclass" not in df.columns and "superclass_idx" in df.columns:
        df["superclass"] = df["superclass_idx"].map(lambda i: CIFAR100_SC_NAMES[i])
    df["dataset"] = "cifar100"
    return df


def load_tiny_imagenet_data():
    """Load or extract Tiny ImageNet per-superclass per-transition data."""
    cached_path = os.path.join(OUTPUT_DIR, "superclass_transition_series_tiny_imagenet.json")

    if os.path.exists(cached_path) and "--force" not in sys.argv:
        print("  Loading cached Tiny ImageNet data...")
        df = pd.read_json(cached_path)
    else:
        print("  Extracting Tiny ImageNet transition data from raw files...")
        def tiny_fine_to_super(class_idx):
            sc = TINY_IDX_TO_SUPER.get(class_idx)
            return sc if sc and sc != "unknown" else None

        rows = extract_transition_data(
            TINY_LANE_ID, TINY_EXP_ID, TINY_LANE_NAME,
            tiny_fine_to_super, TINY_SC_NAMES)

        if not rows:
            print("ERROR: No Tiny ImageNet transition data found.")
            sys.exit(1)

        df = pd.DataFrame(rows)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_json(cached_path, orient="records", indent=2)
        print(f"  Saved: {cached_path}")

    df["dataset"] = "tiny_imagenet"
    return df


def build_panel(df):
    """Build lagged panel from per-superclass per-transition data."""
    rows = []
    for (dataset, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
        group = group.sort_values("transition")
        transitions = group["transition"].values
        ab_h = group["ab_h"].values
        di_h = group["di_h"].values
        tg_h = group["tg_h"].values

        for i in range(len(transitions) - 1):
            rows.append({
                "dataset": dataset,
                "lane": lane,
                "superclass": sc,
                "t": transitions[i],
                "ab_h_t": ab_h[i],
                "di_h_t": di_h[i],
                "tg_h_t": tg_h[i],
                "ab_h_t1": ab_h[i + 1],
                "di_h_t1": di_h[i + 1],
                "tg_h_t1": tg_h[i + 1],
            })
    return pd.DataFrame(rows)


def run_granger_pair(panel, predictor_col, outcome_lag_col, own_history_col, label):
    """Run panel Granger test with dataset + superclass + lane FE."""
    import statsmodels.api as sm

    print(f"\n{'=' * 60}")
    print(f"PANEL GRANGER: {label}")
    print(f"{'=' * 60}")

    y = panel[outcome_lag_col].astype(float).reset_index(drop=True)

    # Fixed effects
    sc_dummies = pd.get_dummies(panel["superclass"], prefix="sc", drop_first=True).astype(float)
    lane_dummies = pd.get_dummies(panel["lane"], prefix="lane", drop_first=True).astype(float)
    dataset_dummies = pd.get_dummies(panel["dataset"], prefix="ds", drop_first=True).astype(float)

    fe = pd.concat([sc_dummies.reset_index(drop=True),
                     lane_dummies.reset_index(drop=True),
                     dataset_dummies.reset_index(drop=True)], axis=1)

    # Restricted: outcome(t+1) ~ outcome(t) + FE
    X_r = pd.concat([panel[[own_history_col]].astype(float).reset_index(drop=True), fe], axis=1)
    X_r = sm.add_constant(X_r)
    model_r = sm.OLS(y, X_r).fit()

    # Full: outcome(t+1) ~ outcome(t) + predictor(t) + FE
    X_f = pd.concat([panel[[own_history_col, predictor_col]].astype(float).reset_index(drop=True), fe], axis=1)
    X_f = sm.add_constant(X_f)
    model_f = sm.OLS(y, X_f).fit()

    # Granger F
    f_stat = ((model_r.ssr - model_f.ssr) / 1) / (model_f.ssr / model_f.df_resid)
    p_value = 1 - f_dist.cdf(f_stat, 1, model_f.df_resid)

    coef = model_f.params[predictor_col]
    tstat = model_f.tvalues[predictor_col]

    print(f"  N observations: {len(panel)}")
    print(f"  Restricted R² ({own_history_col} + FE): {model_r.rsquared:.4f}")
    print(f"  Full R² (+ {predictor_col}):            {model_f.rsquared:.4f}")
    print(f"  ΔR²: {model_f.rsquared - model_r.rsquared:.4f}")
    print(f"  {predictor_col} β = {coef:.4f}, t = {tstat:.2f}")
    print(f"  Granger F = {f_stat:.2f}, p = {p_value:.2e}")

    return {
        "n_obs": len(panel),
        "restricted_r2": round(model_r.rsquared, 4),
        "full_r2": round(model_f.rsquared, 4),
        "delta_r2": round(model_f.rsquared - model_r.rsquared, 4),
        "coef": round(coef, 4),
        "tstat": round(tstat, 4),
        "granger_f": round(f_stat, 4),
        "granger_p": float(f"{p_value:.2e}"),
        "sig": "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns",
        "_model_r": model_r,
        "_model_f": model_f,
        "_f_stat": f_stat,
        "_fe": fe,
    }


def run_per_superclass_granger(panel, predictor_col, outcome_lag_col, own_history_col, label):
    """Per-superclass Granger tests pooling across lanes within each superclass."""
    import statsmodels.api as sm

    print(f"\n{'-' * 40}")
    print(f"Per-superclass: {label}")
    print(f"  {'Superclass':<35} {'N':>4} {'F':>8} {'p':>10} {'p_adj':>10} {'coef':>8}")
    print("  " + "-" * 80)

    all_scs = sorted(panel["superclass"].unique())
    n_tests = len(all_scs)
    results = {}

    for sc in all_scs:
        sc_panel = panel[panel["superclass"] == sc].reset_index(drop=True)
        if len(sc_panel) < 5:
            continue

        lane_dummies = pd.get_dummies(sc_panel["lane"], prefix="lane", drop_first=True).astype(float)

        y_sc = sc_panel[outcome_lag_col].astype(float).reset_index(drop=True)

        X_r = pd.concat([sc_panel[[own_history_col]].astype(float).reset_index(drop=True),
                          lane_dummies.reset_index(drop=True)], axis=1)
        X_r = sm.add_constant(X_r)

        X_f = pd.concat([sc_panel[[own_history_col, predictor_col]].astype(float).reset_index(drop=True),
                          lane_dummies.reset_index(drop=True)], axis=1)
        X_f = sm.add_constant(X_f)

        try:
            m_r = sm.OLS(y_sc, X_r).fit()
            m_f = sm.OLS(y_sc, X_f).fit()
            f_sc = ((m_r.ssr - m_f.ssr) / 1) / (m_f.ssr / m_f.df_resid)
            p_sc = 1 - f_dist.cdf(f_sc, 1, m_f.df_resid)
            p_adj = min(p_sc * n_tests, 1.0)
            coef = m_f.params.get(predictor_col, 0)
            sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else ""
            results[sc] = {"F": round(f_sc, 4), "p": round(p_sc, 6),
                           "p_adj": round(p_adj, 6), "coef": round(coef, 4), "sig": sig,
                           "n": len(sc_panel)}
            print(f"  {sc:<35} {len(sc_panel):>4} {f_sc:>8.3f} {p_sc:>10.6f} {p_adj:>10.4f} {coef:>8.4f} {sig}")
        except Exception as e:
            print(f"  {sc:<35} ERROR: {e}")

    n_sig = sum(1 for v in results.values() if v["sig"])
    n_pos = sum(1 for v in results.values() if v["coef"] > 0)
    print(f"\n  Significant after Bonferroni: {n_sig}/{n_tests}")
    print(f"  Positive coefficient: {n_pos}/{n_tests}")
    return results, n_sig, n_pos


def run_permutation_test(panel, predictor_col, outcome_lag_col, own_history_col,
                         observed_f, label, n_perms=1000):
    """Permutation test: shuffle predictor within each (lane, superclass) panel."""
    import statsmodels.api as sm

    print(f"\n{'-' * 40}")
    print(f"Permutation test ({n_perms} perms): {label}")

    y = panel[outcome_lag_col].astype(float).reset_index(drop=True)

    sc_dummies = pd.get_dummies(panel["superclass"], prefix="sc", drop_first=True).astype(float)
    lane_dummies = pd.get_dummies(panel["lane"], prefix="lane", drop_first=True).astype(float)
    dataset_dummies = pd.get_dummies(panel["dataset"], prefix="ds", drop_first=True).astype(float)
    fe = pd.concat([sc_dummies.reset_index(drop=True), lane_dummies.reset_index(drop=True),
                     dataset_dummies.reset_index(drop=True)], axis=1)

    X_r = pd.concat([panel[[own_history_col]].astype(float).reset_index(drop=True), fe], axis=1)
    X_r = sm.add_constant(X_r)
    model_r = sm.OLS(y, X_r).fit()

    perm_f_stats = []
    for _ in range(n_perms):
        panel_perm = panel.copy()
        for (lane, sc), group in panel_perm.groupby(["lane", "superclass"]):
            idx = group.index
            panel_perm.loc[idx, predictor_col] = np.random.permutation(
                panel_perm.loc[idx, predictor_col].values)

        X_perm = pd.concat([panel_perm[[own_history_col, predictor_col]].astype(float).reset_index(drop=True),
                             fe], axis=1)
        X_perm = sm.add_constant(X_perm)
        model_perm = sm.OLS(y, X_perm).fit()
        f_perm = ((model_r.ssr - model_perm.ssr) / 1) / (model_perm.ssr / model_perm.df_resid)
        perm_f_stats.append(f_perm)

    perm_p = np.mean(np.array(perm_f_stats) >= observed_f)
    print(f"  Observed F: {observed_f:.2f}")
    print(f"  Permutation p: {perm_p:.4f}")
    print(f"  95th percentile null F: {np.percentile(perm_f_stats, 95):.2f}")
    return round(perm_p, 4)


# ── Onset timing analysis ────────────────────────────────────────────────

def compute_onset_metrics(df):
    """For each (dataset, lane, superclass), compute onset/peak timing for each process.

    Metrics per process:
      - peak_t: transition with maximum count
      - onset_t: first transition where count > 10% of that superclass's max
      - center_of_mass: count-weighted mean transition
    """
    rows = []
    for (dataset, lane, sc), group in df.groupby(["dataset", "lane", "superclass"]):
        group = group.sort_values("transition")
        transitions = group["transition"].values.astype(float)

        for process in ["ab_h", "di_h", "tg_h"]:
            counts = group[process].values.astype(float)
            total = counts.sum()

            if total == 0:
                rows.append({"dataset": dataset, "lane": lane, "superclass": sc,
                             "process": process, "peak_t": np.nan,
                             "onset_t": np.nan, "center_of_mass": np.nan,
                             "total_count": 0})
                continue

            peak_t = transitions[np.argmax(counts)]
            threshold = 0.1 * counts.max()
            onset_idx = np.where(counts > threshold)[0]
            onset_t = transitions[onset_idx[0]] if len(onset_idx) > 0 else np.nan
            com = np.average(transitions, weights=counts)

            rows.append({"dataset": dataset, "lane": lane, "superclass": sc,
                         "process": process, "peak_t": peak_t,
                         "onset_t": onset_t, "center_of_mass": com,
                         "total_count": total})

    return pd.DataFrame(rows)


def cross_sectional_onset_analysis(onset_df):
    """Test whether Ab-E onset/peak predicts Di-E onset/peak across superclasses."""
    print(f"\n{'=' * 60}")
    print("ONSET TIMING: Cross-Sectional Correlations")
    print(f"{'=' * 60}")

    results = {}

    for metric in ["peak_t", "onset_t", "center_of_mass"]:
        print(f"\n  Metric: {metric}")

        # Pivot: rows = (dataset, lane, superclass), columns = process
        pivot = onset_df.pivot_table(index=["dataset", "lane", "superclass"],
                                      columns="process", values=metric)
        pivot = pivot.dropna()

        if len(pivot) < 5:
            print(f"    Insufficient data (n={len(pivot)})")
            continue

        for pred, outcome, label in [
            ("ab_h", "di_h", "Ab-E → Di-E"),
            ("tg_h", "ab_h", "Tg-E → Ab-E"),
            ("tg_h", "di_h", "Tg-E → Di-E"),
        ]:
            if pred not in pivot.columns or outcome not in pivot.columns:
                continue
            valid = pivot[[pred, outcome]].dropna()
            if len(valid) < 5:
                continue
            rho, p = spearmanr(valid[pred], valid[outcome])
            print(f"    {label}: Spearman ρ = {rho:.3f}, p = {p:.4f} (n={len(valid)})")
            results[f"{metric}_{pred}_vs_{outcome}"] = {
                "rho": round(rho, 4), "p": round(p, 6), "n": len(valid)}

    return results


def lag_consistency_analysis(onset_df):
    """Compute Ab-E peak → Di-E peak lag per superclass and test consistency."""
    print(f"\n{'=' * 60}")
    print("LAG CONSISTENCY: Ab-E peak → Di-E peak")
    print(f"{'=' * 60}")

    ab_peaks = onset_df[onset_df["process"] == "ab_h"][["dataset", "lane", "superclass", "peak_t"]]
    di_peaks = onset_df[onset_df["process"] == "di_h"][["dataset", "lane", "superclass", "peak_t"]]
    tg_peaks = onset_df[onset_df["process"] == "tg_h"][["dataset", "lane", "superclass", "peak_t"]]

    merged_ab_di = ab_peaks.merge(di_peaks, on=["dataset", "lane", "superclass"],
                                   suffixes=("_ab", "_di"))
    merged_ab_di["lag"] = merged_ab_di["peak_t_di"] - merged_ab_di["peak_t_ab"]

    merged_tg_ab = tg_peaks.merge(ab_peaks, on=["dataset", "lane", "superclass"],
                                   suffixes=("_tg", "_ab"))
    merged_tg_ab["lag"] = merged_tg_ab["peak_t_ab"] - merged_tg_ab["peak_t_tg"]

    merged_tg_di = tg_peaks.merge(di_peaks, on=["dataset", "lane", "superclass"],
                                   suffixes=("_tg", "_di"))
    merged_tg_di["lag"] = merged_tg_di["peak_t_di"] - merged_tg_di["peak_t_tg"]

    results = {}
    for name, merged in [("Ab→Di", merged_ab_di), ("Tg→Ab", merged_tg_ab), ("Tg→Di", merged_tg_di)]:
        lags = merged["lag"].dropna()
        n_positive = (lags > 0).sum()
        n_zero = (lags == 0).sum()
        n_negative = (lags < 0).sum()
        print(f"\n  {name} peak lag:")
        print(f"    Mean: {lags.mean():.2f} ± {lags.std():.2f} transitions")
        print(f"    Median: {lags.median():.1f}")
        print(f"    Positive (outcome after predictor): {n_positive}/{len(lags)}")
        print(f"    Zero (concurrent): {n_zero}/{len(lags)}")
        print(f"    Negative (reversed): {n_negative}/{len(lags)}")

        # Per-dataset breakdown
        for ds in merged["dataset"].unique():
            ds_lags = merged[merged["dataset"] == ds]["lag"].dropna()
            n_pos_ds = (ds_lags > 0).sum()
            print(f"    [{ds}] mean={ds_lags.mean():.2f}, pos={n_pos_ds}/{len(ds_lags)}")

        results[name] = {
            "mean_lag": round(lags.mean(), 4),
            "std_lag": round(lags.std(), 4),
            "median_lag": round(lags.median(), 4),
            "n_positive": int(n_positive),
            "n_zero": int(n_zero),
            "n_negative": int(n_negative),
            "n_total": len(lags),
        }

    return results


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_results(panel, df, onset_df, granger_results, onset_results, lag_results):
    """Create comprehensive figure."""
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # ── Row 1: Panel Granger scatters ────────────────────────────────────
    directions = [
        ("ab_h_t", "di_h_t1", "Ab-E(t) → Di-E(t+1)", "#3B82F6", "ab_to_di"),
        ("tg_h_t", "ab_h_t1", "Tg-E(t) → Ab-E(t+1)", "#F59E0B", "tg_to_ab"),
        ("tg_h_t", "di_h_t1", "Tg-E(t) → Di-E(t+1)", "#10B981", "tg_to_di"),
    ]

    for col_idx, (pred, out, title, color, key) in enumerate(directions):
        ax = fig.add_subplot(gs[0, col_idx])
        x = np.log1p(panel[pred].values)
        y = np.log1p(panel[out].values)
        # Color by dataset
        is_cifar = panel["dataset"].values == "cifar100"
        ax.scatter(x[is_cifar], y[is_cifar], alpha=0.15, s=8, c=color, label="CIFAR-100")
        ax.scatter(x[~is_cifar], y[~is_cifar], alpha=0.3, s=12, c="red", marker="x", label="TinyImageNet")
        z = np.polyfit(x, y, 1)
        p_fn = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p_fn(x_line), "k-", linewidth=1.5)
        res = granger_results[key]
        ax.set_title(f"{title}\nF={res['granger_f']:.1f}, β={res['coef']:.3f}", fontsize=10)
        ax.set_xlabel(f"log({pred.replace('_t', '')})", fontsize=9)
        ax.set_ylabel(f"log({out.replace('_t1', '')})", fontsize=9)
        if col_idx == 0:
            ax.legend(fontsize=7, loc="upper left")

    # Summary table in 4th column
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    table_data = []
    for key, label in [("ab_to_di", "Ab→Di"), ("tg_to_ab", "Tg→Ab"), ("tg_to_di", "Tg→Di")]:
        res = granger_results[key]
        rev = granger_results.get(f"{key}_reverse", {})
        table_data.append([label, f"{res['granger_f']:.1f}",
                           f"{res['coef']:.3f}", f"{res['delta_r2']:.4f}",
                           res["sig"]])
    table = ax.table(cellText=table_data,
                      colLabels=["Direction", "F", "β", "ΔR²", "Sig"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    ax.set_title("Panel Granger Summary", fontsize=11, pad=20)

    # ── Row 2: Onset timing ──────────────────────────────────────────────
    # Panel A: Ab-E peak vs Di-E peak scatter (cross-sectional)
    ax = fig.add_subplot(gs[1, 0])
    pivot_peak = onset_df.pivot_table(index=["dataset", "lane", "superclass"],
                                       columns="process", values="peak_t").dropna()
    if "ab_h" in pivot_peak.columns and "di_h" in pivot_peak.columns:
        valid = pivot_peak[["ab_h", "di_h"]].dropna()
        datasets = valid.index.get_level_values("dataset")
        is_c = datasets == "cifar100"
        ax.scatter(valid["ab_h"][is_c], valid["di_h"][is_c], c="#3B82F6", s=20, alpha=0.6, label="CIFAR-100")
        ax.scatter(valid["ab_h"][~is_c], valid["di_h"][~is_c], c="red", s=30, marker="x", alpha=0.8, label="TinyIN")
        # Identity line
        lims = [min(valid["ab_h"].min(), valid["di_h"].min()) - 0.5,
                max(valid["ab_h"].max(), valid["di_h"].max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        rho, p = spearmanr(valid["ab_h"], valid["di_h"])
        ax.set_xlabel("Ab-E peak transition", fontsize=9)
        ax.set_ylabel("Di-E peak transition", fontsize=9)
        ax.set_title(f"Ab-E vs Di-E peak timing\nρ={rho:.3f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=7)

    # Panel B: Tg-E peak vs Ab-E peak
    ax = fig.add_subplot(gs[1, 1])
    if "tg_h" in pivot_peak.columns and "ab_h" in pivot_peak.columns:
        valid = pivot_peak[["tg_h", "ab_h"]].dropna()
        datasets = valid.index.get_level_values("dataset")
        is_c = datasets == "cifar100"
        ax.scatter(valid["tg_h"][is_c], valid["ab_h"][is_c], c="#F59E0B", s=20, alpha=0.6, label="CIFAR-100")
        ax.scatter(valid["tg_h"][~is_c], valid["ab_h"][~is_c], c="red", s=30, marker="x", alpha=0.8, label="TinyIN")
        lims = [min(valid["tg_h"].min(), valid["ab_h"].min()) - 0.5,
                max(valid["tg_h"].max(), valid["ab_h"].max()) + 0.5]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        rho, p = spearmanr(valid["tg_h"], valid["ab_h"])
        ax.set_xlabel("Tg-E peak transition", fontsize=9)
        ax.set_ylabel("Ab-E peak transition", fontsize=9)
        ax.set_title(f"Tg-E vs Ab-E peak timing\nρ={rho:.3f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=7)

    # Panel C: Peak lag distributions
    ax = fig.add_subplot(gs[1, 2])
    ab_peaks = onset_df[onset_df["process"] == "ab_h"][["dataset", "lane", "superclass", "peak_t"]]
    di_peaks = onset_df[onset_df["process"] == "di_h"][["dataset", "lane", "superclass", "peak_t"]]
    tg_peaks = onset_df[onset_df["process"] == "tg_h"][["dataset", "lane", "superclass", "peak_t"]]

    lag_ab_di = ab_peaks.merge(di_peaks, on=["dataset", "lane", "superclass"],
                                suffixes=("_ab", "_di"))
    lag_ab_di["lag"] = lag_ab_di["peak_t_di"] - lag_ab_di["peak_t_ab"]

    lag_tg_ab = tg_peaks.merge(ab_peaks, on=["dataset", "lane", "superclass"],
                                suffixes=("_tg", "_ab"))
    lag_tg_ab["lag"] = lag_tg_ab["peak_t_ab"] - lag_tg_ab["peak_t_tg"]

    bins = np.arange(-5.5, 6.5, 1)
    ax.hist(lag_ab_di["lag"].dropna(), bins=bins, alpha=0.6, color="#3B82F6",
            label=f"Ab→Di (μ={lag_ab_di['lag'].mean():.1f})", edgecolor="white")
    ax.hist(lag_tg_ab["lag"].dropna(), bins=bins, alpha=0.6, color="#F59E0B",
            label=f"Tg→Ab (μ={lag_tg_ab['lag'].mean():.1f})", edgecolor="white")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Peak lag (transitions)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title("Peak timing lag distribution", fontsize=10)
    ax.legend(fontsize=8)

    # Panel D: Center-of-mass scatter
    ax = fig.add_subplot(gs[1, 3])
    pivot_com = onset_df.pivot_table(index=["dataset", "lane", "superclass"],
                                      columns="process", values="center_of_mass").dropna()
    if "ab_h" in pivot_com.columns and "di_h" in pivot_com.columns:
        valid = pivot_com[["ab_h", "di_h"]].dropna()
        datasets = valid.index.get_level_values("dataset")
        is_c = datasets == "cifar100"
        ax.scatter(valid["ab_h"][is_c], valid["di_h"][is_c], c="#3B82F6", s=20, alpha=0.6, label="CIFAR-100")
        ax.scatter(valid["ab_h"][~is_c], valid["di_h"][~is_c], c="red", s=30, marker="x", alpha=0.8, label="TinyIN")
        lims = [min(valid["ab_h"].min(), valid["di_h"].min()) - 0.2,
                max(valid["ab_h"].max(), valid["di_h"].max()) + 0.2]
        ax.plot(lims, lims, "k--", alpha=0.3, linewidth=1)
        rho, p = spearmanr(valid["ab_h"], valid["di_h"])
        ax.set_xlabel("Ab-E center of mass", fontsize=9)
        ax.set_ylabel("Di-E center of mass", fontsize=9)
        ax.set_title(f"Center-of-mass timing\nρ={rho:.3f}, p={p:.4f}", fontsize=10)
        ax.legend(fontsize=7)

    # ── Row 3: Per-superclass bar charts ─────────────────────────────────
    for col_idx, (key, title, color) in enumerate([
        ("ab_to_di", "Ab-E(t)→Di-E(t+1) per SC", "#3B82F6"),
        ("tg_to_ab", "Tg-E(t)→Ab-E(t+1) per SC", "#F59E0B"),
        ("tg_to_di", "Tg-E(t)→Di-E(t+1) per SC", "#10B981"),
    ]):
        ax = fig.add_subplot(gs[2, col_idx])
        sc_res = granger_results.get(f"{key}_per_sc", {})
        if not sc_res:
            continue
        sc_names = sorted(sc_res.keys())
        coefs = [sc_res[sc]["coef"] for sc in sc_names]
        colors = [color if sc_res[sc]["sig"] else "#D1D5DB" for sc in sc_names]
        y_pos = range(len(sc_names))
        ax.barh(y_pos, coefs, color=colors, height=0.7, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([sc.replace("_", " ")[:25] for sc in sc_names], fontsize=5.5)
        ax.axvline(0, color="black", linewidth=0.5)
        n_sig = sum(1 for v in sc_res.values() if v["sig"])
        ax.set_title(f"{title}\n({n_sig}/{len(sc_names)} sig.)", fontsize=9)
        ax.invert_yaxis()

    # 4th subplot: lag consistency per dataset
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    lag_data = []
    for name, lr in lag_results.items():
        lag_data.append([name, f"{lr['mean_lag']:.2f}±{lr['std_lag']:.2f}",
                         f"{lr['n_positive']}/{lr['n_total']}",
                         f"{lr['median_lag']:.1f}"])
    table = ax.table(cellText=lag_data,
                      colLabels=["Pair", "Mean lag", "Positive", "Median"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)
    ax.set_title("Lag Consistency", fontsize=11, pad=20)

    plt.suptitle("Panel Granger Causality: Per-Superclass Onset Timing\n"
                 "CIFAR-100 (9 lanes × 20 SC) + Tiny ImageNet (1 lane × 28 SC)",
                 fontsize=13, fontweight="bold", y=0.98)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig_path = os.path.join(FIG_DIR, "granger_causality_onset.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"\nFigure saved: {fig_path}")
    plt.close()


def main():
    print("=" * 60)
    print("Panel Granger Causality: Per-Superclass Onset Timing")
    print("CIFAR-100 + Tiny ImageNet")
    print("=" * 60)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    print("\n── Loading CIFAR-100 data ──")
    cifar_df = load_cifar100_data()
    print(f"  {len(cifar_df)} rows, {cifar_df['lane'].nunique()} lanes, "
          f"{cifar_df['superclass'].nunique()} superclasses")

    print("\n── Loading Tiny ImageNet data ──")
    tiny_df = load_tiny_imagenet_data()
    print(f"  {len(tiny_df)} rows, {tiny_df['lane'].nunique()} lanes, "
          f"{tiny_df['superclass'].nunique()} superclasses")

    # Combined
    df = pd.concat([cifar_df, tiny_df], ignore_index=True)
    print(f"\n  Combined: {len(df)} rows, {df['superclass'].nunique()} unique superclasses")

    # ── Step 2: Build panel ───────────────────────────────────────────────
    panel = build_panel(df)
    print(f"\n  Panel: {len(panel)} lagged observations")
    print(f"    Datasets: {panel['dataset'].nunique()}")
    print(f"    Lanes: {panel['lane'].nunique()}")
    print(f"    Superclasses: {panel['superclass'].nunique()}")

    # ── Step 3: Panel Granger tests ───────────────────────────────────────
    all_results = {}

    # Ab-E(t) → Di-E(t+1)
    res = run_granger_pair(panel, "ab_h_t", "di_h_t1", "di_h_t", "Ab-E(t) → Di-E(t+1)")
    perm_p = run_permutation_test(panel, "ab_h_t", "di_h_t1", "di_h_t",
                                   res["_f_stat"], "Ab-E(t) → Di-E(t+1)")
    sc_res, n_sig, n_pos = run_per_superclass_granger(panel, "ab_h_t", "di_h_t1", "di_h_t",
                                                       "Ab-E(t) → Di-E(t+1)")
    # Reverse
    res_rev = run_granger_pair(panel, "di_h_t", "ab_h_t1", "ab_h_t", "Reverse: Di-E(t) → Ab-E(t+1)")

    all_results["ab_to_di"] = {k: v for k, v in res.items() if not k.startswith("_")}
    all_results["ab_to_di"]["permutation_p"] = perm_p
    all_results["ab_to_di_reverse"] = {k: v for k, v in res_rev.items() if not k.startswith("_")}
    all_results["ab_to_di_per_sc"] = sc_res

    # Tg-E(t) → Ab-E(t+1)
    res = run_granger_pair(panel, "tg_h_t", "ab_h_t1", "ab_h_t", "Tg-E(t) → Ab-E(t+1)")
    perm_p = run_permutation_test(panel, "tg_h_t", "ab_h_t1", "ab_h_t",
                                   res["_f_stat"], "Tg-E(t) → Ab-E(t+1)")
    sc_res, n_sig, n_pos = run_per_superclass_granger(panel, "tg_h_t", "ab_h_t1", "ab_h_t",
                                                       "Tg-E(t) → Ab-E(t+1)")
    res_rev = run_granger_pair(panel, "ab_h_t", "tg_h_t1", "tg_h_t", "Reverse: Ab-E(t) → Tg-E(t+1)")

    all_results["tg_to_ab"] = {k: v for k, v in res.items() if not k.startswith("_")}
    all_results["tg_to_ab"]["permutation_p"] = perm_p
    all_results["tg_to_ab_reverse"] = {k: v for k, v in res_rev.items() if not k.startswith("_")}
    all_results["tg_to_ab_per_sc"] = sc_res

    # Tg-E(t) → Di-E(t+1)
    res = run_granger_pair(panel, "tg_h_t", "di_h_t1", "di_h_t", "Tg-E(t) → Di-E(t+1)")
    perm_p = run_permutation_test(panel, "tg_h_t", "di_h_t1", "di_h_t",
                                   res["_f_stat"], "Tg-E(t) → Di-E(t+1)")
    sc_res, n_sig, n_pos = run_per_superclass_granger(panel, "tg_h_t", "di_h_t1", "di_h_t",
                                                       "Tg-E(t) → Di-E(t+1)")
    res_rev = run_granger_pair(panel, "di_h_t", "tg_h_t1", "tg_h_t", "Reverse: Di-E(t) → Tg-E(t+1)")

    all_results["tg_to_di"] = {k: v for k, v in res.items() if not k.startswith("_")}
    all_results["tg_to_di"]["permutation_p"] = perm_p
    all_results["tg_to_di_reverse"] = {k: v for k, v in res_rev.items() if not k.startswith("_")}
    all_results["tg_to_di_per_sc"] = sc_res

    # ── Step 4: Onset timing analysis ─────────────────────────────────────
    onset_df = compute_onset_metrics(df)
    onset_results = cross_sectional_onset_analysis(onset_df)
    all_results["onset_correlations"] = onset_results

    # ── Step 5: Lag consistency ───────────────────────────────────────────
    lag_results = lag_consistency_analysis(onset_df)
    all_results["lag_consistency"] = lag_results

    # ── Step 6: Plot ──────────────────────────────────────────────────────
    plot_results(panel, df, onset_df, all_results, onset_results, lag_results)

    # ── Save ──────────────────────────────────────────────────────────────
    # Strip non-serializable keys
    save_results = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            save_results[k] = {k2: v2 for k2, v2 in v.items() if not k2.startswith("_")}
        else:
            save_results[k] = v

    results_path = os.path.join(OUTPUT_DIR, "granger_causality_onset_results.json")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for direction, fwd_key, rev_key in [
        ("Ab-E → Di-E", "ab_to_di", "ab_to_di_reverse"),
        ("Tg-E → Ab-E", "tg_to_ab", "tg_to_ab_reverse"),
        ("Tg-E → Di-E", "tg_to_di", "tg_to_di_reverse"),
    ]:
        fwd = all_results[fwd_key]
        rev = all_results[rev_key]
        fwd_f = fwd["granger_f"]
        rev_f = rev["granger_f"]
        ratio = fwd_f / rev_f if rev_f > 0 else float("inf")
        print(f"\n  {direction}:")
        print(f"    Forward:  F = {fwd_f:.1f}, β = {fwd['coef']:.4f}, ΔR² = {fwd['delta_r2']:.4f} {fwd['sig']}")
        print(f"    Reverse:  F = {rev_f:.1f}, β = {rev['coef']:.4f}")
        print(f"    F ratio (fwd/rev): {ratio:.1f}×")
        print(f"    Permutation p: {fwd.get('permutation_p', 'N/A')}")

    print("\n  Onset timing correlations:")
    for k, v in onset_results.items():
        print(f"    {k}: ρ = {v['rho']:.3f}, p = {v['p']:.4f}")

    print("\n  Lag consistency:")
    for name, lr in lag_results.items():
        print(f"    {name}: mean={lr['mean_lag']:.2f}, "
              f"positive={lr['n_positive']}/{lr['n_total']}, "
              f"median={lr['median_lag']:.1f}")


if __name__ == "__main__":
    main()
