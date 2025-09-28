import argparse
import os
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib as mpl

def configure_matplotlib_fonts(base_size: int = 16):
    """
    Set larger, consistent font sizes across plots.
    """
    mpl.rcParams.update({
        "font.size": base_size,               # default text
        "axes.titlesize": base_size + 2,     # axes title
        "axes.labelsize": base_size,         # x/y labels
        "xtick.labelsize": base_size - 2,    # tick labels
        "ytick.labelsize": base_size - 2,
        "legend.fontsize": base_size - 2,    # legend text
        "figure.titlesize": base_size + 4,   # suptitle
        "savefig.bbox": "tight",             # avoid cutting anything
        "savefig.pad_inches": 0.2,           # small outer padding
    })

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _success_rate_by_label_and_N(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["agent_label", "N"])["success"].mean().reset_index(name="success_rate")
    grp["success_rate"] = grp["success_rate"] * 100.0
    return grp.sort_values(["agent_label", "N"])


def _success_rate_by_model_and_N(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["model_depth", "model_hidden", "split", "N"])["success"].mean().reset_index(name="success_rate")
    grp["success_rate"] = grp["success_rate"] * 100.0
    grp[["model_depth", "model_hidden"]] = grp[["model_depth", "model_hidden"]].astype(int)
    return grp.sort_values(["model_depth", "model_hidden", "split", "N"])


def _overall_success_for_model(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return float(df["success"].mean() * 100.0)


def pick_best_model(full_df: pd.DataFrame) -> Tuple[int, int, pd.DataFrame]:
    models = full_df[full_df["agent"] == "model"].dropna(subset=["model_depth", "model_hidden"])
    if models.empty:
        raise RuntimeError("No model rows found in CSV.")

    best_key = None
    best_tuple = None  # (-eval_succ, -train_succ, depth, hidden)
    for (d, h), _ in models.groupby(["model_depth", "model_hidden"]):
        sub_eval = models[(models["model_depth"] == d) & (models["model_hidden"] == h) & (models["split"] == "eval-Ns")]
        sub_train = models[(models["model_depth"] == d) & (models["model_hidden"] == h) & (models["split"] == "train-Ns")]
        eval_succ = _overall_success_for_model(sub_eval)
        train_succ = _overall_success_for_model(sub_train)
        key = (-eval_succ, -train_succ, int(d), int(h))
        if (best_tuple is None) or (key < best_tuple):
            best_tuple = key
            best_key = (int(d), int(h))

    d, h = best_key
    best_df = models[(models["model_depth"] == d) & (models["model_hidden"] == h)]
    return d, h, best_df


def _split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    x = df[df["split"] == split].copy()
    if x.empty:
        raise RuntimeError(f"No rows for split={split}")
    return x


def plot_best_vs_baselines_success(df: pd.DataFrame, out_pdf: str):
    df_train = _split_df(df, "train-Ns")
    df_eval  = _split_df(df, "eval-Ns")

    base_train = _success_rate_by_label_and_N(df_train[df_train["agent"] == "baseline"])
    base_eval  = _success_rate_by_label_and_N(df_eval[df_eval["agent"] == "baseline"])

    best_d, best_h, best_both = pick_best_model(df)
    tag = f"model(d{best_d},h{best_h})"
    best_train_curve = _success_rate_by_label_and_N(
        best_both[best_both["split"] == "train-Ns"].assign(agent_label=lambda x: tag))
    best_eval_curve = _success_rate_by_label_and_N(
        best_both[best_both["split"] == "eval-Ns"].assign(agent_label=lambda x: tag))

    fig, ax = plt.subplots(figsize=(9, 5))

    for label, sub in base_train.groupby("agent_label"):
        ax.plot(sub["N"], sub["success_rate"], marker="o", linestyle="-", label=f"{label} (train)")
    for label, sub in base_eval.groupby("agent_label"):
        ax.plot(sub["N"], sub["success_rate"], marker="o", linestyle="--", label=f"{label} (eval)")

    ax.plot(best_train_curve["N"], best_train_curve["success_rate"], marker="s", linestyle="-", label=f"{tag} (train)")
    ax.plot(best_eval_curve["N"], best_eval_curve["success_rate"], marker="s", linestyle="--", label=f"{tag} (eval)")

    ax.set_xlabel("N")
    ax.set_ylabel("Success from random starts (%)")
    ax.set_title("Best model vs. baselines")
    ax.grid(True, alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=max(3, len(labels)//2))
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(out_pdf, format='pdf')
    plt.close(fig)

    base_train.assign(split="train-Ns").to_csv(os.path.splitext(out_pdf)[0].replace(".pdf", "_baselines_success_curves_train.csv"), index=False)
    base_eval.assign(split="eval-Ns").to_csv(os.path.splitext(out_pdf)[0].replace(".pdf", "_baselines_success_curves_eval.csv"), index=False)
    best_train_curve.assign(split="train-Ns").to_csv(os.path.splitext(out_pdf)[0].replace(".pdf", "_best_model_success_curve_train.csv"), index=False)
    best_eval_curve.assign(split="eval-Ns").to_csv(os.path.splitext(out_pdf)[0].replace(".pdf", "_best_model_success_curve_eval.csv"), index=False)

    return best_d, best_h


def plot_grid_search_heatmaps(df: pd.DataFrame, out_pdf: str):
    mdf = df[(df["agent"] == "model") & df["model_depth"].notna() & df["model_hidden"].notna()].copy()
    if mdf.empty:
        raise RuntimeError("No model rows found for grid search heatmaps.")

    def summarize(split: str) -> pd.DataFrame:
        sub = mdf[mdf["split"] == split]
        rows = []
        for (d, h), g in sub.groupby(["model_depth", "model_hidden"]):
            rows.append({"depth": int(d), "hidden": int(h), f"{split}_success": float(g["success"].mean() * 100.0)})
        return pd.DataFrame(rows)

    tdf = summarize("train-Ns")
    edf = summarize("eval-Ns")
    summ = pd.merge(tdf, edf, on=["depth", "hidden"], how="outer").fillna(0.0)
    summ["gap_pp"] = summ["eval-Ns_success"] - summ["train-Ns_success"]

    out_csv = os.path.splitext(out_pdf)[0].replace(".pdf", "_summary.csv")
    summ.sort_values(["eval-Ns_success", "train-Ns_success"], ascending=[False, False]).to_csv(out_csv, index=False)

    def _pivot(col: str) -> pd.DataFrame:
        p = summ.pivot(index="depth", columns="hidden", values=col).sort_index().sort_index(axis=1)
        return p

    A = _pivot("train-Ns_success")
    B = _pivot("eval-Ns_success")
    C = _pivot("gap_pp")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    def _heat(ax, mat: pd.DataFrame, title: str, cbar_label: str):
        im = ax.imshow(mat.values, aspect="auto", origin="upper", interpolation="nearest")
        ax.set_xticks(np.arange(len(mat.columns))); ax.set_xticklabels(mat.columns.astype(int))
        ax.set_yticks(np.arange(len(mat.index)));   ax.set_yticklabels(mat.index.astype(int))
        ax.set_xlabel("hidden"); ax.set_ylabel("depth"); ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label(cbar_label)
        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                ax.text(j, i, f"{mat.values[i, j]:.1f}", ha="center", va="center", fontsize=8)

    _heat(axes[0], A, "Train success (%)", "success %")
    _heat(axes[1], B, "Eval success (%)",  "success %")
    _heat(axes[2], C, "Eval − Train (pp)", "pp")

    fig.suptitle("Grid search: success from random starts (train vs eval and gap)", y=1.02)
    fig.savefig(out_pdf, bbox_inches="tight", format='pdf')
    plt.close(fig)


def _rank_models_by_eval_success(df_models: pd.DataFrame, only_depths: Optional[List[int]], only_hiddens: Optional[List[int]], top_k: int):
    sub = df_models.copy()
    if only_depths:
        sub = sub[sub["model_depth"].isin(only_depths)]
    if only_hiddens:
        sub = sub[sub["model_hidden"].isin(only_hiddens)]
    # Rank by overall eval success
    ranks = []
    for (d, h), g in sub[sub["split"] == "eval-Ns"].groupby(["model_depth", "model_hidden"]):
        ranks.append({"depth": int(d), "hidden": int(h), "eval_success": float(g["success"].mean() * 100.0)})
    r = pd.DataFrame(ranks).sort_values("eval_success", ascending=False).head(top_k)
    return r[["depth", "hidden"]].values.tolist()  # list of (d,h)


def plot_grid_search_lines_topK(df: pd.DataFrame, out_pdf: str, top_k: int = 8,
                                only_depths: Optional[List[int]] = None,
                                only_hiddens: Optional[List[int]] = None):
    """
    Line plots vs N for the top-K (depth,hidden) configs by eval success.
    Shows train (solid) and eval (dashed) for each selected model.
    Legend sits above the axes.
    """
    mdf = df[(df["agent"] == "model") & df["model_depth"].notna() & df["model_hidden"].notna()].copy()
    if mdf.empty:
        raise RuntimeError("No model rows found for grid search lines.")

    mdf[["model_depth", "model_hidden"]] = mdf[["model_depth", "model_hidden"]].astype(int)
    picks = _rank_models_by_eval_success(mdf, only_depths, only_hiddens, top_k=top_k)
    if not picks:
        raise RuntimeError("No models matched filters for line plots.")

    curves = _success_rate_by_model_and_N(mdf)
    tag = lambda d, h: f"d{d},h{h}"

    fig, ax = plt.subplots(figsize=(10, 6))

    # Cycle through distinct markers to help readability
    mk = ["o", "s", "^", "D", "v", "P", "X", "h", "*", "p"]
    for idx, (d, h) in enumerate(picks):
        mtrain = curves[(curves["model_depth"] == d) & (curves["model_hidden"] == h) & (curves["split"] == "train-Ns")]
        meval  = curves[(curves["model_depth"] == d) & (curves["model_hidden"] == h) & (curves["split"] == "eval-Ns")]
        marker = mk[idx % len(mk)]
        ax.plot(mtrain["N"], mtrain["success_rate"], marker=marker, linestyle="-",  label=f"{tag(d,h)} (train)")
        ax.plot(meval["N"],  meval["success_rate"],  marker=marker, linestyle="--", label=f"{tag(d,h)} (eval)")

    ax.set_xlabel("N")
    ax.set_ylabel("Success from random starts (%)")
    ax.set_title(f"Grid search (top-{top_k}) — success vs N")
    ax.grid(True, alpha=0.35)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend().remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.2),
               ncol=min(4, len(labels)//2))
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    fig.savefig(out_pdf, format='pdf')
    plt.close(fig)


def plot_grid_search_lines_by_depth_hidden(df: pd.DataFrame,
                                           out_dir: str,
                                           only_depths: Optional[List[int]] = None,
                                           only_hiddens: Optional[List[int]] = None,
                                           do_by_depth: bool = False,
                                           do_by_hidden: bool = False):
    """
    Optional small-multiple line plots grouped by depth or hidden.
    Each figure shows several curves (hidden or depth sweeps) with train/eval styles.
    """
    mdf = df[(df["agent"] == "model") & df["model_depth"].notna() & df["model_hidden"].notna()].copy()
    if mdf.empty:
        return
    mdf[["model_depth", "model_hidden"]] = mdf[["model_depth", "model_hidden"]].astype(int)
    curves = _success_rate_by_model_and_N(mdf)

    if do_by_depth:
        depths = sorted(mdf["model_depth"].unique())
        if only_depths:
            depths = [d for d in depths if d in only_depths]
        for d in depths:
            sub_h = sorted(mdf[mdf["model_depth"] == d]["model_hidden"].unique())
            if only_hiddens:
                sub_h = [h for h in sub_h if h in only_hiddens]
            if not sub_h:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for h in sub_h:
                mtrain = curves[(curves["model_depth"] == d) & (curves["model_hidden"] == h) & (curves["split"] == "train-Ns")]
                meval  = curves[(curves["model_depth"] == d) & (curves["model_hidden"] == h) & (curves["split"] == "eval-Ns")]
                ax.plot(mtrain["N"], mtrain["success_rate"], marker="o", linestyle="-",  label=f"h{h} (train)")
                ax.plot(meval["N"],  meval["success_rate"],  marker="o", linestyle="--", label=f"h{h} (eval)")
            ax.set_xlabel("N"); ax.set_ylabel("Success from random starts (%)")
            ax.set_title(f"Success vs N — depth {d} (solid=train, dashed=eval)")
            ax.grid(True, alpha=0.35)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend().remove()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=max(3, len(labels)//2))
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(os.path.join(out_dir, f"grid_search_lines_by_depth_{d}.pdf"), format='pdf')
            plt.close(fig)

    if do_by_hidden:
        hiddens = sorted(mdf["model_hidden"].unique())
        if only_hiddens:
            hiddens = [h for h in hiddens if h in only_hiddens]
        for h in hiddens:
            sub_d = sorted(mdf[mdf["model_hidden"] == h]["model_depth"].unique())
            if only_depths:
                sub_d = [d for d in sub_d if d in only_depths]
            if not sub_d:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            for d in sub_d:
                mtrain = curves[(curves["model_depth"] == d) & (curves["model_hidden"] == h) & (curves["split"] == "train-Ns")]
                meval  = curves[(curves["model_depth"] == d) & (curves["model_hidden"] == h) & (curves["split"] == "eval-Ns")]
                ax.plot(mtrain["N"], mtrain["success_rate"], marker="o", linestyle="-",  label=f"d{d} (train)")
                ax.plot(meval["N"],  meval["success_rate"],  marker="o", linestyle="--", label=f"d{d} (eval)")
            ax.set_xlabel("N"); ax.set_ylabel("Success from random starts (%)")
            ax.set_title(f"Success vs N — hidden {h} (solid=train, dashed=eval)")
            ax.grid(True, alpha=0.35)
            handles, labels = ax.get_legend_handles_labels()
            plt.legend().remove()
            fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=max(3, len(labels)//2))
            fig.tight_layout(rect=[0, 0, 1, 0.92])
            fig.savefig(os.path.join(out_dir, f"grid_search_lines_by_hidden_{h}.pdf"), dpi=200)
            plt.close(fig)


def main():
    configure_matplotlib_fonts(base_size=20)  # <— add this
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV produced by the evaluation script.")
    ap.add_argument("--out_dir", default="results/plots_success", help="Output directory for plots and CSVs.")

    # Line plot controls
    ap.add_argument("--top_k", type=int, default=8, help="Top-K (depth,hidden) configs by eval success to plot as lines.")
    ap.add_argument("--only_depths", type=int, nargs="*", default=None, help="Optional filter: only these depths.")
    ap.add_argument("--only_hiddens", type=int, nargs="*", default=None, help="Optional filter: only these hiddens.")
    ap.add_argument("--by_depth", action="store_true", help="Also emit per-depth line plots.")
    ap.add_argument("--by_hidden", action="store_true", help="Also emit per-hidden line plots.")

    args = ap.parse_args()

    ensure_dir(args.out_dir)
    df = pd.read_csv(args.csv)

    needed = {"agent", "agent_label", "split", "N", "success", "model_depth", "model_hidden"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"CSV missing columns: {missing}")

    # Best vs baselines (success)
    out_best = os.path.join(args.out_dir, "best_vs_baselines_success.pdf")
    best_d, best_h = plot_best_vs_baselines_success(df, out_best)

    # Grid-search heatmaps
    out_heat = os.path.join(args.out_dir, "grid_search_success.pdf")
    plot_grid_search_heatmaps(df, out_heat)

    # NEW: Grid-search line plots (top-K)
    out_lines = os.path.join(args.out_dir, "grid_search_lines_topK.pdf")
    plot_grid_search_lines_topK(
        df, out_lines, top_k=args.top_k,
        only_depths=args.only_depths, only_hiddens=args.only_hiddens
    )

    # Optional per-depth / per-hidden small multiples
    if args.by_depth or args.by_hidden:
        plot_grid_search_lines_by_depth_hidden(
            df, args.out_dir,
            only_depths=args.only_depths, only_hiddens=args.only_hiddens,
            do_by_depth=args.by_depth, do_by_hidden=args.by_hidden
        )

    print(f"Best model by eval success: depth={best_d}, hidden={best_h}")
    print("Saved:")
    print(f"  - {out_best}")
    print(f"  - {out_heat}")
    print(f"  - {out_lines}")
    if args.by_depth:
        print("  - grid_search_lines_by_depth_*.pdf")
    if args.by_hidden:
        print("  - grid_search_lines_by_hidden_*.pdf")


if __name__ == "__main__":
    main()
