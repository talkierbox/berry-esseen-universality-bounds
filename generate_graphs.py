"""
Fast d-regular graph generator with optional Rust backend.
"""

from pathlib import Path
import os
import numpy as np
import scipy.sparse as sp
from numba import jit
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import List, Dict, Any

# Backend availability
try:
    import fast_graph_gen
    RUST = True
    print("Using Rust backend")
except ImportError:
    RUST = False
    print("Rust backend not found, using Python/Numba")

# Config
BASE          = Path.cwd()
DATA          = BASE / "data"
DATA.mkdir(exist_ok=True)
D_SELECTION   = [3, 5]
N_SELECTION   = [1_000, 10_000, 100_000, 500_000, 1_000_000]
NUM_TO_GEN    = 300
RANDOM_SEED   = 42

# Helpers
def _as_uint8_array(data: Any) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8).copy()
    return np.asarray(data, dtype=np.uint8)

def _to_csr(n: int, rows: Any, cols: Any, data: Any) -> sp.csr_matrix:
    idx_dtype = np.int64 if n > np.iinfo(np.int32).max else np.int32
    return sp.csr_matrix(
        (_as_uint8_array(data),
         (np.asarray(rows, idx_dtype), np.asarray(cols, idx_dtype))),
        shape=(n, n), dtype=np.uint8
    )

# Pure‑Python path (only reasonable for small n)
@jit(nopython=True, cache=True)
def _initial_regular(n: int, d: int) -> np.ndarray:
    A = np.zeros((n, n), np.uint8)
    for i in range(n):
        for k in range(1, d // 2 + 1):
            A[i, (i + k) % n] = A[i, (i - k) % n] = 1
    if d % 2:
        for i in range(n // 2):
            A[i, i + n // 2] = A[i + n // 2, i] = 1
    return A

@jit(nopython=True, cache=True)
def _mix(A: np.ndarray, swaps: int, seed: int) -> None:
    np.random.seed(seed)
    n = A.shape[0]
    edges = np.empty((n * n // 2, 2), np.int32)
    cnt = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                edges[cnt] = (i, j)
                cnt += 1
    edges = edges[:cnt]

    for _ in range(swaps):
        if cnt < 2:
            break
        i, j = np.random.randint(0, cnt, 2)
        if i == j:
            continue
        a, b = edges[i]
        c, d = edges[j]
        if len({a, b, c, d}) < 4:
            continue
        if np.random.rand() < 0.5:
            u1, v1, u2, v2 = a, c, b, d
        else:
            u1, v1, u2, v2 = a, d, b, c
        if u1 == v1 or u2 == v2 or A[u1, v1] or A[u2, v2]:
            continue
        A[a, b] = A[b, a] = A[c, d] = A[d, c] = 0
        A[u1, v1] = A[v1, u1] = A[u2, v2] = A[v2, u2] = 1
        edges[i] = (min(u1, v1), max(u1, v1))
        edges[j] = (min(u2, v2), max(u2, v2))

def _python_graph(n: int, d: int, seed: int, mix: int = 3) -> sp.csr_matrix:
    swaps = mix * n * d
    A = _initial_regular(n, d)
    _mix(A, swaps, seed)
    return sp.csr_matrix(A)

# Rust wrappers
def _rust_single_csr(n: int, d: int, seed: int, mix: int = 3) -> sp.csr_matrix:
    rows, cols, data = fast_graph_gen.generate_uniform_regular(n, d, seed, mix)
    return _to_csr(n, rows, cols, data)

def _rust_batch_csr(n: int, d: int, seeds: List[int], mix: int = 3) -> List[sp.csr_matrix]:
    res = fast_graph_gen.generate_multiple_graphs_sparse(n, d, len(seeds), seeds[0], mix)
    csr_list = []
    for seed, (rows, cols, data) in zip(seeds, res):
        csr_list.append(_to_csr(n, rows, cols, data))
    return csr_list

# Disk IO
def _save_csr(A: sp.csr_matrix, path: Path) -> None:
    sp.save_npz(path, A, compressed=True)

def _build_and_save(n: int, d: int, seed: int, out: Path, use_rust: bool = True) -> str:
    if (n * d) & 1:
        raise ValueError("n * d must be even")
    if use_rust and RUST:
        A = _rust_single_csr(n, d, seed)
    else:
        if n > 10_000:
            raise MemoryError("Python path too slow for large n")
        A = _python_graph(n, d, seed)
    fname = out / f"g_d{d}_n{n}_{seed}.npz"
    _save_csr(A, fname)
    return str(fname)

def generate_graphs(
    d: int, 
    n_vals: List[int], 
    samples: int, 
    out_dir: Path, 
    base_seed: int = 42, 
    batch: int = 256
) -> Dict[int, List[str]]:
    rng = np.random.default_rng(base_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = {}
    for n in n_vals:
        seeds = rng.integers(0, 2**32 - 1, size=samples, dtype=np.uint32)
        written = []
        if RUST:
            batch = 64 if n > 50_000 else 256 if n > 10_000 else 512
            for s0 in tqdm(range(0, samples, batch), desc=f"d={d} n={n}", unit="graph", ncols=90):
                chunk = seeds[s0:s0 + batch].tolist()
                for A, seed in zip(_rust_batch_csr(n, d, chunk), chunk):
                    fname = out_dir / f"g_d{d}_n{n}_{seed}.npz"
                    _save_csr(A, fname)
                    written.append(str(fname))
        else:
            backend = "loky"
            paths = Parallel(n_jobs=os.cpu_count(), backend=backend)(
                delayed(_build_and_save)(n, d, int(s), out_dir, False) for s in tqdm(seeds, desc=f"d={d} n={n}")
            )
            written.extend(paths)
        idx[n] = written
    return idx

if __name__ == "__main__":
    for d in D_SELECTION:
        out = DATA / f"d{d}"
        generate_graphs(d, N_SELECTION, NUM_TO_GEN, out, RANDOM_SEED)
        print(f"Finished d={d} -> {out}")
