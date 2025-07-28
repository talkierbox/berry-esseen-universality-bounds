from __future__ import annotations
import os
import random
from pathlib import Path
from typing import Sequence, List
import numpy as np
import igraph as ig
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm.auto import tqdm

BASE          = Path(os.getcwd())
DATA          = BASE / "data"
DATA.mkdir(exist_ok=True)

RANDOM_SEED   = 42
NUM_TO_GEN    = 10_000                # per (d, n)
D_SELECTION   = [3, 5]
N_SELECTION   = [1_000, 100_000, 1_000_000]

def _build_and_store(n: int, d: int, seed: int, out_dir: Path) -> str:
    """Generate one d-regular graph, save its adjacency as CSR .npz."""
    if (n * d) & 1:
        raise ValueError("n x d must be even for a d-regular graph")

    random.seed(seed)
    g = ig.Graph.K_Regular(n, d, directed=False, multiple=False)

    # igraph → SciPy CSR in C; O(m) memory
    A: sp.csr_matrix = g.get_adjacency_sparse().tocsr()
    fname = out_dir / f"g_d{d}_n{n}_{seed}.npz"
    sp.save_npz(fname, A, compressed=True)          # 2‑3 bytes/NZ on disk
    return str(fname)

def generate_regular_graphs(
    d: int, n_list: Sequence[int], samples: int, out_dir: Path,
    base_seed: int = 42, chunk: int = 1_024, n_jobs: int | None = None,
) -> dict[int, List[str]]:
    """Generate `samples` random d-regular graphs for every n in `n_list`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_jobs = os.cpu_count() if n_jobs is None else n_jobs
    rng    = np.random.default_rng(base_seed)

    index: dict[int, List[str]] = {}
    for n in n_list:
        seeds   = rng.integers(0, 2**32 - 1, size=samples, dtype=np.uint32)
        written = []

        # keep memory head‑room for the largest graphs
        chunk = min(chunk, 128) if n > 100_000 else chunk

        with tqdm(total=samples, desc=f"d={d}, n={n}", unit="graph") as bar:
            for s0 in range(0, samples, chunk):
                s1     = min(s0 + chunk, samples)
                paths  = Parallel(
                    n_jobs=n_jobs, backend="loky", batch_size=8
                )(delayed(_build_and_store)(n, d, int(s), out_dir)
                  for s in seeds[s0:s1])
                written.extend(paths)
                bar.update(s1 - s0)
        index[n] = written
    return index

if __name__ == "__main__":
    for d in tqdm(D_SELECTION, desc="Overall progress"):
        out_d = DATA / f"d{d}"
        out_d.mkdir(parents=True, exist_ok=True)
        generate_regular_graphs(
            d=d, n_list=N_SELECTION, samples=NUM_TO_GEN,
            out_dir=out_d, base_seed=RANDOM_SEED,
            chunk=512, n_jobs=None,
        )
        print(f"finished d={d} → {out_d}")