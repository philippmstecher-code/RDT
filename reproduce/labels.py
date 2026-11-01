"""Names used in the Source Data workbooks for the short names the analysis code writes.

The workbooks give runs, arms, conditions and measures in the words of the figure legends; the CSVs that
reproduce/figures.sh writes keep the code's short names. reproduce/check.py applies the same renaming before it
compares a rebuilt table with the workbook.
"""
import re

COLUMNS = {"lane": "run", "lane_prefix": "run_id"}

_SEEDLESS = {
    "gradsc": "coarse block", "gradjoint": "matched block", "gradenergy": "energy-matched block",
    "gradrand": "random directions", "gradrg": "random-grouping block", "sham": "no block",
    "superclass": "superclass directions", "random": "random directions", "randgroup": "random grouping",
    "ab": "superclass-selective features", "ctrl": "matched features",
    "between": "noise across superclasses", "within": "noise within superclasses", "clean": "clean training",
    "matched": "noise across, matched similarity",
    "sc_true": "true superclasses", "true": "true superclasses", "sc_true_K20": "true superclasses",
    "sc_rand": "random grouping", "rand": "random grouping", "sc_rand_K20": "random grouping",
    "sc_blind": "superclass-blind", "blind": "superclass-blind", "sc_blind_K20": "superclass-blind",
    "scratch_E30": "scratch",
}
ARMS = {**_SEEDLESS,
        "gradsc minus gradenergy": "coarse block minus energy-matched block",
        "gradsc minus gradjoint": "coarse block minus matched block",
        "scratch minus sham": "scratch minus no block"}
VALUES = {
    "arm": ARMS,
    "control": ARMS,
    "contrast": {"gradsc - gradenergy": "coarse block - energy-matched block",
                 "gradrg - gradenergy": "random-grouping block - energy-matched block"},
    "condition": {**{f"switch_e{e:02d}": f"{e} superclass epochs" for e in (5, 10, 20, 25, 30)},
                  "switch_e15": "11 superclass epochs (resumed run)",
                  "standard50": "0 superclass epochs", "standard": "clean training",
                  "between_sc_p03": "noise across superclasses, p = 0.3", "random_p03": "random noise, p = 0.3"},
    "measure": {"fresh_assignment": "superclass assignment (fresh read-out)", "val_accuracy": "fine-class accuracy",
                "net_assignment": "superclass accuracy (network)", "net_fine": "fine-class accuracy (network)",
                "oracle_logits": "class within the true superclass",
                "d_assignment_pp": "superclass assignment change (pp)", "d_fine_pp": "fine-class accuracy change (pp)",
                "d_class_given_true_superclass_pp": "class given the true superclass, change (pp)",
                "d_superclass_accuracy_pp": "superclass accuracy change (pp)",
                "acc_0_10": "fine-class accuracy, mean of epochs 0-10", "acc_30": "fine-class accuracy, epoch 30",
                "probe_30": "superclass probe, epoch 30", "probe_5": "superclass probe, epoch 5"},
    "group": {"core": "main runs", "indep_init": "independent SAE initialisation", "tinyimagenet": "Tiny ImageNet"},
    "dataset": {"cifar100": "CIFAR-100", "tiny_imagenet": "Tiny ImageNet"},
    "family": {"curriculum_vs_sham": "no-block runs (clean reference)"},
    "window": {"last_block_1_10": "last block, epochs 1-10"},
    "panel": {"last_block_1_10": "last block, epochs 1-10"},
}
_TABLE = ((r"\bbetween - within\b", "across - within"),
          (r"\bdelay_to_30pct_accuracy\b", "epochs to 30% accuracy"),
          (r"\bfine_acc_epoch30\b", "fine-class accuracy at epoch 30"),
          (r"\bfine_acc_epochs0_10\b", "fine-class accuracy, mean of epochs 0-10"),
          (r"\bprobe_epoch30\b", "superclass probe at epoch 30"))


def value(column, v):
    """The workbook's name for value v of a column (unchanged if it has none)."""
    if not isinstance(v, str):
        return v
    if column == "table":
        for a, b in _TABLE:
            v = re.sub(a, b, v)
        return v
    return VALUES.get(column, {}).get(v, v)


def column(name):
    return COLUMNS.get(name, name)


def relabel(columns, rows):
    """(columns, rows) of a CSV table in the workbook's names."""
    return [column(c) for c in columns], [[value(c, v) for c, v in zip(columns, r)] for r in rows]
