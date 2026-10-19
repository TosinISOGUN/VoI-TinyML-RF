"""End-to-end Phase-1 computational experiment (Fig. 2 workflow)."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import models as M
from . import plots
from .config import EXP, SEED, VOI
from .data import build_splits
from .evaluation import bootstrap_ci, classification_report, exact_mcnemar
from .telemetry import PrecomputedLink, evaluate, plan_b0, plan_b1, plan_b2, plan_b3
from .voi import Components, fit_novelty_scale, novelty, raw_novelty

# Published values, for side-by-side comparison in the summary.
PAPER = {
    "models": {"Decision Tree": (84.48, 81.97), "Random Forest": (89.85, 88.08),
               "MLP": (83.58, 78.68), "Tiny 1-D CNN": (82.69, 78.59)},
    "comm": {"B0": (335, 10720, 15823.54, 70.78, 76.41), "B1": (335, 4020, 9257.76, 70.78, 76.41),
             "B2": (177, 3684, 6420.16, 70.59, 76.41), "B3": (92, 2764, 10738.93, 94.44, 87.04)},
    "thresholds": {"tau_D": 0.90, "T1": 0.575, "T2": 0.675, "q50": 0.3716, "q80": 1.3221},
}


def train_models(s):
    tr, va, te = s.train, s.val, s.test
    out = []
    for name, model in [("Decision Tree", M.decision_tree()), ("Random Forest", M.random_forest()),
                        ("MLP", M.mlp())]:
        model.fit(tr.X_tab, tr.y)
        out.append(dict(model=name, est=model, view="tab", size_kb=M.sklearn_size_kb(model)))
    cnn = M.TinyCNNClassifier().fit(tr.X_seq, tr.y, va.X_seq, va.y)
    out.append(dict(model="Tiny 1-D CNN", est=cnn, view="seq", size_kb=cnn.size_kb()))
    for m in out:
        for part, ds in (("val", va), ("test", te)):
            X = ds.X_tab if m["view"] == "tab" else ds.X_seq
            m[f"proba_{part}"] = m["est"].predict_proba(X)
            m[f"pred_{part}"] = m[f"proba_{part}"].argmax(axis=1)
        m["val_macro_f1"] = classification_report(va.y, m["pred_val"])["macro_f1"]
        m.update(classification_report(te.y, m["pred_test"]))
        m["ci"] = bootstrap_ci(te.y, m["pred_test"])
    return out


def calibrate_thresholds(P, D, critical):
    """Validation grid search: minimize energy s.t. CEDR >= 0.90 and PDR >= 0.82 (Sec. 4.8)."""
    links = [PrecomputedLink(len(P), 5000 + i) for i in range(VOI.calibration_realizations)]
    rows = []
    for T1, T2 in itertools.product(VOI.grid, VOI.grid):
        if T2 < T1:
            continue
        plan = plan_b3(P, D, T1, T2)
        r = [evaluate(plan, l, critical) for l in links]
        rows.append(dict(T1=T1, T2=T2, packets=int(plan.transmit.sum()),
                         energy_mj=np.mean([q["energy_mj"] for q in r]),
                         pdr=np.mean([q["pdr"] if q["packets"] else 0.0 for q in r]), cedr=np.mean([q["cedr"] for q in r])))
    grid = pd.DataFrame(rows)
    feasible = grid[(grid.cedr >= VOI.min_cedr) & (grid.pdr >= VOI.min_pdr)]
    pick = (feasible if len(feasible) else grid).sort_values(["energy_mj", "T1", "T2"]).iloc[0]
    return float(pick.T1), float(pick.T2), grid


def summarize(runs):
    df = pd.DataFrame(runs)
    out = {"packets": int(df.packets.iloc[0]), "bytes": int(df.bytes.iloc[0])}
    for k in ("energy_mj", "pdr", "cedr"):
        out[k] = float(df[k].mean())
        out[f"{k}_p2.5"], out[f"{k}_p97.5"] = (float(v) for v in np.percentile(df[k], [2.5, 97.5]))
    return out


def run(out_dir: str | Path = "results", n_realizations: int = EXP.n_realizations, verbose: bool = True):
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    # 1-3: data, temporal windows, chronological split
    s = build_splits()
    log(f"Sequences: {len(s.train.y) + len(s.val.y) + len(s.test.y)} "
        f"(train {len(s.train.y)}, val {len(s.val.y)}, test {len(s.test.y)})")
    log(f"Class thresholds q0.50={s.thresholds[0]:.4f} q0.80={s.thresholds[1]:.4f}; "
        f"test classes {np.bincount(s.test.y).tolist()}")

    # 4-5: candidate models, evaluation and selection
    models = train_models(s)
    best = max(models, key=lambda m: m["val_macro_f1"])  # selection on validation, not test
    yte = s.test.y
    mcnemar = {m["model"]: exact_mcnemar(yte, best["pred_test"], m["pred_test"])
               for m in models if m is not best}
    pd.DataFrame([{
        "model": m["model"], "accuracy_pct": 100 * m["accuracy"],
        "macro_precision_pct": 100 * m["macro_precision"], "macro_recall_pct": 100 * m["macro_recall"],
        "macro_f1_pct": 100 * m["macro_f1"], "artifact_kb": m["size_kb"],
        "acc_ci_low": 100 * m["ci"]["accuracy"][0], "acc_ci_high": 100 * m["ci"]["accuracy"][1],
        "f1_ci_low": 100 * m["ci"]["macro_f1"][0], "f1_ci_high": 100 * m["ci"]["macro_f1"][1],
        "val_macro_f1_pct": 100 * m["val_macro_f1"],
    } for m in models]).round(2).to_csv(out / "tables/table2_models.csv", index=False)
    pd.DataFrame(best["per_class"]).assign(
        precision=lambda d: 100 * d.precision, recall=lambda d: 100 * d.recall, f1=lambda d: 100 * d.f1
    ).round(2).to_csv(out / "tables/table3_selected_model_per_class.csv", index=False)
    for m in models:
        log(f"  {m['model']:<14} acc {100 * m['accuracy']:.2f}%  macro-F1 {100 * m['macro_f1']:.2f}%  "
            f"size {m['size_kb']:.2f} KB")
    log(f"Selected model (best validation macro-F1): {best['model']}")

    # 6: VoI controller calibration (training novelty scale, validation thresholds)
    x = s.series.to_numpy()
    tau_d = fit_novelty_scale(x, s.train.t_index)
    D_val = novelty(raw_novelty(x, s.val.t_index), tau_d)
    D_test = novelty(raw_novelty(x, s.test.t_index), tau_d)
    T1, T2, grid = calibrate_thresholds(best["proba_val"], D_val, s.val.y == 2)
    grid.round(4).to_csv(out / "tables/voi_threshold_grid_validation.csv", index=False)
    log(f"VoI calibration: tau_D={tau_d:.2f}  T1={T1:.3f}  T2={T2:.3f}")

    # 7-8: B0-B3 under Monte Carlo link realizations
    P, crit, n = best["proba_test"], yte == 2, len(yte)
    plans = [plan_b0(n), plan_b1(n), plan_b2(best["pred_test"]), plan_b3(P, D_test, T1, T2)]
    links = [PrecomputedLink(n, EXP.mc_seed_base + i) for i in range(n_realizations)]
    runs = {p.name: [evaluate(p, l, crit) for l in links] for p in plans}
    comm = [dict(system=name, **summarize(r)) for name, r in runs.items()]
    pd.DataFrame(comm).round(4).to_csv(out / "tables/table4_5_communication_montecarlo.csv", index=False)
    pd.concat([pd.DataFrame(r).assign(realization=np.arange(len(r))) for r in runs.values()]).to_csv(
        out / "tables/montecarlo_realizations.csv", index=False)

    # Ablation on the fixed link trace (seed 202)
    link202 = PrecomputedLink(n, EXP.ablation_link_seed)
    variants = [
        ("Full B3", Components(), True), ("Without uncertainty", Components(uncertainty=False), True),
        ("Without novelty", Components(novelty=False), True), ("Without information age", Components(age=False), True),
        ("Without severity", Components(severity=False), True), ("Without link adaptation", Components(), False),
    ]
    ablation = []
    for name, comp, adaptive in variants:
        r = evaluate(plan_b3(P, D_test, T1, T2, comp, adaptive, name), link202, crit)
        ablation.append(dict(variant=name, **{k: r[k] for k in (
            "packets", "bytes", "energy_mj", "pdr", "cedr", "delivered", "critical_delivered", "critical_total")}))
    pd.DataFrame(ablation).round(4).to_csv(out / "tables/table6_ablation_seed202.csv", index=False)

    # Figures 3-6
    plots.fig_models(models, out / "figures/fig3_model_comparison.png")
    plots.fig_confusion(best["confusion_matrix"], out / "figures/fig4_confusion_matrix.png")
    plots.fig_communication(comm, out / "figures/fig5_communication.png")
    plots.fig_ablation(ablation, out / "figures/fig6_ablation.png")

    b0, b3 = comm[0], comm[3]
    headline = {
        "transmission_reduction_pct": 100 * (1 - b3["packets"] / b0["packets"]),   # Eq. 23
        "byte_reduction_pct": 100 * (1 - b3["bytes"] / b0["bytes"]),
        "energy_reduction_pct": 100 * (1 - b3["energy_mj"] / b0["energy_mj"]),     # Eq. 24
        "pdr_gain_pp": 100 * (b3["pdr"] - b0["pdr"]),
        "cedr_gain_pp": 100 * (b3["cedr"] - b0["cedr"]),
    }
    results = {
        "seed": SEED, "n_realizations": n_realizations,
        "dataset": {"n_train": len(s.train.y), "n_val": len(s.val.y), "n_test": len(yte),
                    "thresholds": s.thresholds, "test_class_counts": np.bincount(yte).tolist()},
        "models": [{k: v for k, v in m.items() if k in (
            "model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "size_kb",
            "val_macro_f1", "ci", "per_class", "confusion_matrix")} for m in models],
        "selected_model": best["model"], "mcnemar_p": mcnemar,
        "voi": {"tau_D": tau_d, "T1": T1, "T2": T2},
        "communication": comm, "headline": headline, "ablation": ablation, "paper": PAPER,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))

    log("\nCommunication (packets/bytes fixed; energy/PDR/CEDR = Monte Carlo means):")
    for r in comm:
        log(f"  {r['system']:<30} {r['packets']:>4} pkts {r['bytes']:>6} B  "
            f"{r['energy_mj']:>9.2f} mJ  PDR {100 * r['pdr']:.2f}%  CEDR {100 * r['cedr']:.2f}%")
    log("Headline: " + ", ".join(f"{k}={v:.2f}" for k, v in headline.items()))
    log(f"\nWrote {out / 'results.json'}, tables/ and figures/")
    return results
