"""Train baseline + LightGBM, tune thresholds, write artifacts to models/.

Usage: python -m fraud.train [--data data/creditcard.csv.zip] [--trials 30]
"""

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud.drift import build_reference
from fraud.features import FEATURES, V_COLS, to_matrix
from fraud.threshold import confusion_at, pick_thresholds, sweep

FN_COST, FP_COST, TARGET = 120.0, 5.0, 0.90
PR_AUC_FLOOR = 0.70

# LightGBM 4.6 + sklearn 1.6 quirk: harmless warning on every numpy predict.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def passes_gate(ap: float, baseline_ap: float) -> bool:
    """Catch broken models: must reach the floor and beat the linear baseline."""
    return ap >= PR_AUC_FLOOR and ap > baseline_ap


def time_split(df: pd.DataFrame):
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    a, b = int(len(df) * 0.70), int(len(df) * 0.85)
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


def xy(df: pd.DataFrame):
    return to_matrix(df.to_dict("records")), df["Class"].to_numpy()


def tune_lgbm(Xtr, ytr, Xva, yva, trials: int) -> lgb.LGBMClassifier:
    max_spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    def fit(params: dict) -> lgb.LGBMClassifier:
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            subsample_freq=1,
            metric="average_precision",
            random_state=42,
            verbose=-1,
            **params,
        )
        model.fit(
            Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        return model

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, max_spw, log=True),
        }
        model = fit(params)
        return average_precision_score(yva, model.predict_proba(Xva)[:, 1])

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    print(f"best val PR-AUC={study.best_value:.4f} params={study.best_params}")
    return fit(study.best_params)


# n_train=2000: a 500-row pool re-sampled 500x gave 5% false "watch" alarms (PSI noise floor).
def sample_rows(train, test, n_train=2000, n_test=400) -> pd.DataFrame:
    """UI demo rows. 'train': legit rows across all hours (looks like the drift reference).
    'test': unseen legit + all frauds from the last (night-time) period, for honest scoring."""
    cols = V_COLS + ["Amount", "Class"]
    parts = [
        train[train.Class == 0].sample(n_train, random_state=42).assign(source="train"),
        test[test.Class == 0].sample(n_test, random_state=42).assign(source="test"),
        test[test.Class == 1].assign(source="test"),
    ]
    return pd.concat(parts)[cols + ["source"]].round(6)


def policy_report(y, scores, t: float) -> dict:
    tp, fp, fn, tn = confusion_at(y, scores, t)
    return {
        "threshold": t,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "cost": fn * FN_COST + fp * FP_COST,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/creditcard.csv.zip")
    p.add_argument("--out", default="models")
    p.add_argument("--samples", default="ui/samples.csv")
    p.add_argument("--trials", type=int, default=30)
    args = p.parse_args(argv)

    df = pd.read_csv(args.data)
    train, val, test = time_split(df)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = xy(train), xy(val), xy(test)
    print(
        f"rows train/val/test={len(ytr)}/{len(yva)}/{len(yte)} "
        f"frauds={ytr.sum()}/{yva.sum()}/{yte.sum()}"
    )

    base = make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(Xtr, ytr)
    base_ap = float(average_precision_score(yte, base.predict_proba(Xte)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    model = tune_lgbm(Xtr, ytr, Xva, yva, args.trials)
    s_val, s_te = model.predict_proba(Xva)[:, 1], model.predict_proba(Xte)[:, 1]
    ap = float(average_precision_score(yte, s_te))
    print(f"test PR-AUC baseline={base_ap:.4f} lightgbm={ap:.4f}")
    if not passes_gate(ap, base_ap):
        print(
            f"FAIL: test PR-AUC {ap:.4f} needs >= {PR_AUC_FLOOR} and > baseline {base_ap:.4f}; "
            "artifacts not written",
            file=sys.stderr,
        )
        return 1

    picked = pick_thresholds(yva, s_val, FN_COST, FP_COST, TARGET)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    model.booster_.save_model(str(out / "model.txt"))

    config = {
        "model_version": datetime.now(UTC).strftime("%Y%m%d-%H%M%S"),
        "features": FEATURES,
        "thresholds": picked["thresholds"],
        "warnings": picked["warnings"],
        "costs": {"fn": FN_COST, "fp": FP_COST},
        "target": TARGET,
        "expected_flag_rate": float(np.mean(s_val >= picked["thresholds"]["cost"])),
        "metrics": {
            "test_pr_auc": ap,
            "test_roc_auc": float(roc_auc_score(yte, s_te)),
            "baseline_test_pr_auc": base_ap,
            "test": {k: policy_report(yte, s_te, t) for k, t in picked["thresholds"].items()},
        },
    }
    prec, rec, _ = precision_recall_curve(yva, s_val)
    step = max(1, len(prec) // 300)
    curves = {
        "pr": {"precision": prec[::step].tolist(), "recall": rec[::step].tolist()},
        "sweep": sweep(yva, s_val, FN_COST, FP_COST),
    }
    # Feature reference from train; score reference from val (train scores are overfit).
    reference = build_reference(Xtr, FEATURES, s_val)

    for name, obj in [("config", config), ("curves", curves), ("reference", reference)]:
        (out / f"{name}.json").write_text(json.dumps(obj, indent=2))

    sample_rows(train, test).to_csv(args.samples, index=False)

    print(json.dumps(config["metrics"]["test"], indent=2))
    for w in picked["warnings"]:
        print(f"WARNING: {w}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
