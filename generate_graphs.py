"""
Generate and save random d-regular graphs using igraph.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, List

import numpy as np
import igraph as ig  # switched from networkx
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import random

BASE = Path(os.path.abspath(""))
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

RANDOM_SEED = 42
NUM_TO_GENERATE = 2_000

D_SELECTION = [3, 5, 10, 20] # 20 > N^0.25 for N = 100_000 --- breaks the bound
N_SELECTION = [5_000, 10_000, 50_000, 100_000]

def _build_and_store(n: int, d: int, seed: int, out_dir: Path) -> str:
    """Build one d-regular graph and save its edge list."""
    if (n * d) % 2:
        raise ValueError("n * d must be even")
    
    random.seed(seed)
    g_ig = ig.Graph.K_Regular(n, d, directed=False, multiple=False)

    edges = np.asarray(g_ig.get_edgelist(), dtype=np.uint32)
    fname = out_dir / f"g_d{d}_n{n}_{seed}.npz"
    np.savez_compressed(fname, edges=edges)
    return str(fname)


def generate_regular_graphs(
    *,
    d: int,
    n_list: Sequence[int],
    samples: int,
    out_dir: str | Path,
    base_seed: int = 42,
    chunk_size: int = 1_024,
    n_jobs: int | None = None,
) -> dict[int, List[str]]:
    """Generate `samples` random d-regular graphs for each n in `n_list`."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_jobs = os.cpu_count() if n_jobs is None else n_jobs
    rng = np.random.default_rng(base_seed)

    index: dict[int, List[str]] = {}

    for n in n_list:
        seeds = rng.integers(0, 2**32 - 1, size=samples, dtype=np.uint32)
        saved: List[str] = []

        with tqdm(total=samples, desc=f"d={d}, n={n}", unit="graph") as bar:
            for start in range(0, samples, chunk_size):
                stop = min(start + chunk_size, samples)
                paths = Parallel(n_jobs=n_jobs, backend="loky", batch_size=8)(
                    delayed(_build_and_store)(n, d, int(s), out_root)
                    for s in seeds[start:stop]
                )
                saved.extend(paths)
                bar.update(stop - start)

        index[n] = saved

    return index


if __name__ == "__main__":
    print("Generating graphs... This may take a while.")

    for d in tqdm(D_SELECTION, desc="Overall progress"):
        out_dir = DATA / f"d{d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        generate_regular_graphs(
            d=d,
            n_list=N_SELECTION,
            samples=NUM_TO_GENERATE,
            out_dir=out_dir,
            base_seed=RANDOM_SEED,
            chunk_size=1024,
            n_jobs=None,
        )
        print(f"Finished d={d}. Graphs saved in {out_dir}")
