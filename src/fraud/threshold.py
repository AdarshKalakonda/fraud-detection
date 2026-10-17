"""Pick decision thresholds from validation scores by cost, recall or precision."""

import numpy as np

# Fine steps: scores from a well-calibrated fraud model mostly sit below 0.05.
GRID = np.arange(1, 1000) / 1000


def confusion_at(y, scores, t: float) -> tuple[int, int, int, int]:
    pred = scores >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def sweep(y, scores, fn_cost: float, fp_cost: float) -> list[dict]:
    rows = []
    for t in GRID:
        tp, fp, fn, tn = confusion_at(y, scores, t)
        rows.append(
            {
                "threshold": float(t),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "cost": fn * fn_cost + fp * fp_cost,
            }
        )
    return rows


def pick_thresholds(y, scores, fn_cost=120.0, fp_cost=5.0, target=0.90) -> dict:
    rows = sweep(y, scores, fn_cost, fp_cost)
    warnings = []

    # Lowest cost; on ties prefer the higher threshold (fewer alerts).
    cost = min(rows, key=lambda r: (r["cost"], -r["threshold"]))

    ok = [r for r in rows if r["recall"] >= target]
    if ok:
        recall = max(ok, key=lambda r: r["threshold"])
    else:
        recall = max(rows, key=lambda r: (r["recall"], r["threshold"]))
        warnings.append(f"recall {target} unreachable; best is {recall['recall']:.3f}")

    ok = [r for r in rows if r["precision"] >= target]
    if ok:
        precision = min(ok, key=lambda r: r["threshold"])
    else:
        precision = max(rows, key=lambda r: (r["precision"], -r["threshold"]))
        warnings.append(f"precision {target} unreachable; best is {precision['precision']:.3f}")

    return {
        "thresholds": {
            "cost": cost["threshold"],
            "recall": recall["threshold"],
            "precision": precision["threshold"],
        },
        "warnings": warnings,
    }
