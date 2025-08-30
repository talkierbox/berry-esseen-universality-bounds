"""
Fast d-regular graph generator with Rust backend (batch), optimized I/O and CSR build.
"""

from pathlib import Path
import os
import numpy as np
import scipy.sparse as sp
from numba import jit
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple

try:
    from rand_d_regular import d_regular_near_uniform, d_regular_near_uniform_batch  # type: ignore
    RUST = True
    HAS_BATCH = True
    print("Using Rust backend (batch): rand_d_regular")
except ImportError:
    try:
        from rand_d_regular import d_regular_near_uniform  # type: ignore
        RUST = True
        HAS_BATCH = False
        print("Using Rust backend (single): rand_d_regular")
    except ImportError:
        RUST = False
        HAS_BATCH = False
        print("Rust backend not found, using Python/Numba")

# CONFIG

BASE        = Path.cwd()
DATA        = BASE / "data"
DATA.mkdir(exist_ok=True)
D_SELECTION = [3, 5, 8]
N_SELECTION = [1_000, 10_000, 100_000, 500_000, 1_000_000]
NUM_TO_GEN  = 2000
RANDOM_SEED = 42

# ----

def _as_uint8_array(data: Any) -> np.ndarray:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return np.frombuffer(data, dtype=np.uint8).copy()
    return np.asarray(data, dtype=np.uint8)

def _to_csr(n: int, rows: Any, cols: Any, data: Any) -> sp.csr_matrix:
    # For our n (<= 1e6) int32 indices are fine and smaller/faster than int64
    idx_dtype = np.int32
    return sp.csr_matrix(
        (_as_uint8_array(data),
         (np.asarray(rows, idx_dtype), np.asarray(cols, idx_dtype))),
        shape=(n, n), dtype=bool  # store adjacency as boolean
    )

def _edges_to_csr(n: int, edges_uv: np.ndarray) -> sp.csr_matrix:
    """
    edges_uv shape (m,2) with undirected unique edges (u < v).
    Build symmetric CSR with ones (bool dtype). Minimize allocations.
    """
    # Force contiguous int32 views without copy if possible
    u = np.asarray(edges_uv[:, 0], dtype=np.int32, order="C")
    v = np.asarray(edges_uv[:, 1], dtype=np.int32, order="C")
    m = u.size
    m2 = m * 2

    # Pre-allocate row/col arrays once
    r = np.empty(m2, dtype=np.int32)
    c = np.empty(m2, dtype=np.int32)
    r[:m] = u;  c[:m] = v
    r[m:] = v;  c[m:] = u

    data = np.ones(m2, dtype=bool)  # bool compresses better on disk later
    return sp.csr_matrix((data, (r, c)), shape=(n, n), dtype=bool)

@jit(nopython=True, cache=True)
def _initial_regular(n: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a simple d-regular graph (circulant + optional perfect matching)
    and return (A, edges). A is (n,n) uint8 adjacency. edges is (m,2) int32 with u<v.
    """
    A = np.zeros((n, n), np.uint8)
    m = (n * d) // 2
    edges = np.empty((m, 2), np.int32)
    cnt = 0

    half = d // 2
    for i in range(n):
        for k in range(1, half + 1):
            j1 = (i + k) % n
            if i < j1 and A[i, j1] == 0:
                A[i, j1] = 1; A[j1, i] = 1
                edges[cnt, 0] = i; edges[cnt, 1] = j1; cnt += 1
            j2 = (i - k) % n
            if i < j2 and A[i, j2] == 0:
                A[i, j2] = 1; A[j2, i] = 1
                edges[cnt, 0] = i; edges[cnt, 1] = j2; cnt += 1

    if d % 2 == 1:
        for i in range(n // 2):
            j = i + n // 2
            if A[i, j] == 0:
                A[i, j] = 1; A[j, i] = 1
                edges[cnt, 0] = i; edges[cnt, 1] = j; cnt += 1

    return A, edges

@jit(nopython=True, cache=True)
def _mix(A: np.ndarray, edges: np.ndarray, swaps: int, seed: int) -> None:
    """
    Degree-preserving double-edge swaps on adjacency A and edge list 'edges' (u<v).
    """
    np.random.seed(seed)
    m = edges.shape[0]
    if m < 2:
        return

    for _ in range(swaps):
        i = np.random.randint(0, m)
        j = np.random.randint(0, m - 1)
        if j >= i:
            j += 1
        a, b = edges[i, 0], edges[i, 1]
        c, d = edges[j, 0], edges[j, 1]
        if a == c or a == d or b == c or b == d:
            continue

        if np.random.random() < 0.5:
            u1, v1, u2, v2 = a, c, b, d
        else:
            u1, v1, u2, v2 = a, d, b, c

        if u1 == v1 or u2 == v2:
            continue
        if A[u1, v1] or A[u2, v2]:
            continue

        A[a, b] = 0; A[b, a] = 0
        A[c, d] = 0; A[d, c] = 0

        A[u1, v1] = 1; A[v1, u1] = 1
        A[u2, v2] = 1; A[v2, u2] = 1

        if u1 > v1:
            t = u1; u1 = v1; v1 = t
        if u2 > v2:
            t2 = u2; u2 = v2; v2 = t2
        edges[i, 0] = u1; edges[i, 1] = v1
        edges[j, 0] = u2; edges[j, 1] = v2

def _python_graph(n: int, d: int, seed: int, mix: int = 3) -> sp.csr_matrix:
    """
    Slow path for small n: build circulant d-regular graph and perform swaps.
    swaps ≈ mix * m * log n where m = n*d/2.
    """
    swaps = max(1, int(mix * (n * d // 2) * max(1.0, np.log(max(2, n)))))  # ~ c * m * log n
    A, edges = _initial_regular(n, d)
    _mix(A, edges, swaps, seed)
    return _edges_to_csr(n, edges)

# Rust

def _rust_single_csr(n: int, d: int, seed: int, mix: float = 5.0) -> sp.csr_matrix:
    """
    Build one CSR using the rand_d_regular Rust backend.
    Returns (m,2) undirected edges; we symmetrize to CSR.
    """
    edges = d_regular_near_uniform(n, d, int(seed), float(mix))  # (m,2) uint32
    return _edges_to_csr(n, np.asarray(edges))

def _rust_batch_csr(n: int, d: int, seeds: List[int], mix: float = 5.0) -> List[sp.csr_matrix]:
    """
    Generate multiple graphs using the Rust batch API if available; else per-call fallback.
    """
    if HAS_BATCH:
        # Returns list of (m_i,2) arrays
        arrays = d_regular_near_uniform_batch(n, d, [int(s) for s in seeds], float(mix))
        return [_edges_to_csr(n, np.asarray(arr)) for arr in arrays]
    else:
        return [_rust_single_csr(n, d, int(s), mix) for s in seeds]

# I/O

def _save_csr(A: sp.csr_matrix, path: Path) -> None:
    # Uncompressed for speed during generation. Compress/shard offline if needed.
    sp.save_npz(path, A, compressed=False)

def _build_and_save(n: int, d: int, seed: int, out: Path, use_rust: bool = True, mix: float = 5.0) -> str:
    if (n * d) & 1:
        raise ValueError("n * d must be even")
    if d >= n:
        raise ValueError("require d < n")

    if use_rust and RUST:
        A = _rust_single_csr(n, d, seed, mix=mix)
    else:
        if n > 10_000:
            raise MemoryError("Python/Numba path too slow for large n; enable Rust backend")
        A = _python_graph(n, d, seed, mix=int(max(1, mix)))

    fname = out / f"g_d{d}_n{n}_{seed}.npz"
    _save_csr(A, fname)
    return str(fname)

def generate_graphs(
    d: int,
    n_vals: List[int],
    samples: int,
    out_dir: Path,
    base_seed: int = 42,
    batch: int = 256,
    mix: float = 5.0,
    parallel_backend: str = "loky",
) -> Dict[int, List[str]]:
    """
    Generate multiple d-regular graphs for each n in n_vals.

    Parameters
    ----------
    d : int
        Regular degree (must satisfy d < n and n*d even).
    n_vals : List[int]
        Different vertex counts to generate.
    samples : int
        Number of graphs per n.
    out_dir : Path
        Output directory. Files saved as NPZ CSR (bool dtype).
    base_seed : int
        RNG seed for reproducible seed streams.
    batch : int
        Max batch size per chunk (used for Rust batch).
    mix : float
        Swap-iteration multiplier for double-edge swaps.
        Effective K ≈ mix * m * log n, where m = n*d/2. Default 5.0.
    parallel_backend : str
        joblib backend for Python fallback.

    Returns
    -------
    Dict[int, List[str]]
        Mapping n -> list of written file paths.
    """
    rng = np.random.default_rng(base_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx: Dict[int, List[str]] = {}

    for n in n_vals:
        if (n * d) & 1:
            raise ValueError(f"n * d must be even (got n={n}, d={d})")
        if d >= n:
            raise ValueError(f"require d < n (got n={n}, d={d})")

        seeds = rng.integers(0, 2**32 - 1, size=samples, dtype=np.uint32)
        written: List[str] = []

        if RUST:
            # Light auto-tune: smaller batches for larger n to keep memory in check.
            local_batch = batch
            if n > 500_000:
                local_batch = min(batch, 16)
            elif n > 100_000:
                local_batch = min(batch, 32)
            elif n > 50_000:
                local_batch = min(batch, 64)
            elif n > 10_000:
                local_batch = min(batch, 256)
            else:
                local_batch = min(batch, 512)

            for s0 in tqdm(range(0, samples, local_batch), desc=f"d={d} n={n}", unit="graph", ncols=90):
                chunk = seeds[s0:s0 + local_batch].astype(np.uint32).tolist()
                csrs = _rust_batch_csr(n, d, chunk, mix=mix)
                for A, seed in zip(csrs, chunk):
                    fname = out_dir / f"g_d{d}_n{n}_{int(seed)}.npz"
                    _save_csr(A, fname)
                    written.append(str(fname))
        else:
            # Python/Numba fallback in parallel (small n only)
            if n > 10_000:
                raise MemoryError("Python path too slow for large n; please enable a Rust backend")
            paths = Parallel(n_jobs=os.cpu_count(), backend=parallel_backend)(
                delayed(_build_and_save)(n, d, int(s), out_dir, False, mix)
                for s in tqdm(seeds, desc=f"d={d} n={n}")
            )
            written.extend(paths)

        idx[n] = written

    return idx

if __name__ == "__main__":
    for d in D_SELECTION:
        out = DATA / f"d{d}"
        generate_graphs(d, N_SELECTION, NUM_TO_GEN, out, RANDOM_SEED)
        print(f"Finished d={d} -> {out}")
