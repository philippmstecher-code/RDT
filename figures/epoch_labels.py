#!/usr/bin/env python3
"""Shared utility for dual-line epoch tick labels across figure scripts."""


def format_epoch_ticks(ax, epochs, axis='x', primary_labels=None, every_n=1, show_every_n=None):
    """Add epoch annotations as secondary tick labels.

    Creates two-line tick labels:
        0         1         2         3
      (ep.0)   (ep.1)    (ep.2)    (ep.3)

    Args:
        ax: matplotlib Axes
        epochs: list of epoch values per tick position
        axis: 'x' or 'y'
        primary_labels: optional override for primary tick labels (default: 0,1,2,...)
        every_n: show epoch label every N ticks (for crowded axes)
        show_every_n: only place ticks every N positions (reduces tick density).
                      If None, all positions get ticks.
    """
    n = len(epochs)
    if primary_labels is None:
        primary_labels = list(range(n))

    if show_every_n is not None:
        positions = list(range(0, n, show_every_n))
    else:
        positions = list(range(n))

    dual_labels = []
    for idx, pos in enumerate(positions):
        primary = str(primary_labels[pos])
        if idx % every_n == 0:
            dual_labels.append(f"{primary}\n(ep.{epochs[pos]})")
        else:
            dual_labels.append(primary)

    if axis == 'x':
        ax.set_xticks(positions)
        ax.set_xticklabels(dual_labels, rotation=45, ha='right')
    else:
        ax.set_yticks(positions)
        ax.set_yticklabels(dual_labels)
