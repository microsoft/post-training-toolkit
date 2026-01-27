
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def plot_series(df: pd.DataFrame, x: str, ys: List[str], title: str, outfile: Path,
                ylabel: Optional[str] = None, legend_loc: str = "best") -> None:
    _ensure_dir(outfile.parent)
    plt.figure(figsize=(10, 4))
    sns.set_style("whitegrid")
    for y in ys:
        if y in df.columns:
            plt.plot(df[x], df[y], label=y)
    plt.title(title)
    plt.xlabel(x)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend(loc=legend_loc)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_reward(df: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "reward.png"
    plot_series(df, x="step", ys=["reward_mean", "reward_std"], title="Reward (mean/std) over steps", outfile=out,
                ylabel="reward")
    return out

def plot_kl(df: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "kl.png"
    plot_series(df, x="step", ys=["kl"], title="KL divergence over steps", outfile=out, ylabel="KL")
    return out

def plot_drift(df: pd.DataFrame, outdir: Path, cosine_key: str = "embedding_cosine_to_sft") -> Path:
    out = outdir / "drift.png"
    plot_series(df, x="step", ys=[cosine_key], title="Policy drift (cosine to SFT) over steps", outfile=out,
                ylabel="cosine")
    return out

def plot_slices(df: pd.DataFrame, outdir: Path, slice_prefix: str = "slice:") -> Optional[Path]:
    slice_cols = [c for c in df.columns if c.startswith(slice_prefix)]
    if not slice_cols:
        return None
    out = outdir / "slices.png"
    plot_series(df, x="step", ys=slice_cols, title="Slice metrics over steps", outfile=out, ylabel="score")
    return out

