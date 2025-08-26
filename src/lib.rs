use hashbrown::HashSet;
use numpy::{IntoPyArray, PyArray2};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rayon::prelude::*;

/// Insert an undirected simple edge (u,v) (stored as (min,max)).
#[inline]
fn add_edge(u: usize, v: usize, adj: &mut [HashSet<usize>], edges: &mut Vec<(usize, usize)>) {
    debug_assert!(u != v);
    let (a, b) = if u < v { (u, v) } else { (v, u) };
    edges.push((a, b));
    adj[u].insert(v);
    adj[v].insert(u);
}

#[inline]
fn has_edge(u: usize, v: usize, adj: &[HashSet<usize>]) -> bool {
    adj[u].contains(&v)
}

/// Build a trivial simple d-regular graph on n vertices:
/// circulant with neighbors i +/- 1..d/2 (mod n), plus one perfect matching if d is odd.
/// Returns (sorted unique edge list, adjacency sets).
fn initial_d_regular(n: usize, d: usize) -> (Vec<(usize, usize)>, Vec<HashSet<usize>>) {
    assert!(d < n, "require d < n");
    assert!((n * d) % 2 == 0, "n*d must be even");

    let mut edges = Vec::with_capacity(n * d / 2);
    let mut adj: Vec<HashSet<usize>> = (0..n).map(|_| HashSet::with_capacity(d)).collect();

    let half = d / 2;
    for i in 0..n {
        for k in 1..=half {
            let j1 = (i + k) % n;
            if !has_edge(i, j1, &adj) {
                add_edge(i, j1, &mut adj, &mut edges);
            }
            let j2 = (n + i - k) % n;
            if !has_edge(i, j2, &adj) {
                add_edge(i, j2, &mut adj, &mut edges);
            }
        }
    }

    if d % 2 == 1 {
        // perfect matching between i and i + n/2 (requires even n)
        assert!(n % 2 == 0, "odd d requires even n");
        for i in 0..(n / 2) {
            let j = i + n / 2;
            if !has_edge(i, j, &adj) {
                add_edge(i, j, &mut adj, &mut edges);
            }
        }
    }

    // Deduplicate/sort for safety (circulant pushes can repeat)
    edges.sort_unstable();
    edges.dedup();
    (edges, adj)
}

/// Attempt a double-edge swap (a-b, c-d) -> (a-c, b-d) or (a-d, b-c) uniformly.
/// Returns true on success (graph remains simple).
fn try_swap(
    e_idx1: usize,
    e_idx2: usize,
    edges: &mut Vec<(usize, usize)>,
    adj: &mut [HashSet<usize>],
    rng: &mut StdRng,
) -> bool {
    let (a, b) = edges[e_idx1];
    let (c, d) = edges[e_idx2];
    if a == c || a == d || b == c || b == d {
        return false; // distinct endpoints
    }

    // Random pairing
    let do_ax = rng.gen_bool(0.5);
    let (mut u1, mut v1, mut u2, mut v2) = if do_ax { (a, c, b, d) } else { (a, d, b, c) };

    if u1 == v1 || u2 == v2 {
        return false;
    }
    if has_edge(u1, v1, adj) || has_edge(u2, v2, adj) {
        return false;
    }

    // Remove old edges
    adj[a].remove(&b);
    adj[b].remove(&a);
    adj[c].remove(&d);
    adj[d].remove(&c);

    // Add new edges
    add_edge(u1, v1, adj, edges);
    add_edge(u2, v2, adj, edges);

    // Remove old entries from edges vec (replace-with-last trick).
    // Remove the higher index first to keep the lower index valid.
    let i_hi = e_idx1.max(e_idx2);
    let i_lo = e_idx1.min(e_idx2);
    edges.swap_remove(i_hi);
    edges.swap_remove(i_lo);

    true
}

/// Perform k random valid double-edge swaps (best-effort).
fn randomize_with_swaps(
    edges: &mut Vec<(usize, usize)>,
    adj: &mut [HashSet<usize>],
    k: usize,
    rng: &mut StdRng,
) {
    let m0 = edges.len();
    if m0 < 2 || k == 0 {
        return;
    }
    for _ in 0..k {
        let m_now = edges.len();
        if m_now < 2 {
            break;
        }
        let e1 = rng.gen_range(0..m_now);
        let mut e2 = rng.gen_range(0..m_now);
        if e2 == e1 {
            e2 = (e2 + 1) % m_now;
        }
        let _ = try_swap(e1, e2, edges, adj, rng);
    }
}

/// Core generator for a single graph. Returns edge list (u<v).
fn generate_one(n: usize, d: usize, seed: u64, iters_mul: f64) -> Vec<(usize, usize)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut edges, mut adj) = initial_d_regular(n, d);

    let m = edges.len();
    let k = ((iters_mul * (m as f64) * (n as f64).ln()).ceil() as usize).max(1);
    randomize_with_swaps(&mut edges, &mut adj, k, &mut rng);

    edges
}

#[pyfunction]
fn d_regular_near_uniform<'py>(
    py: Python<'py>,
    n: usize,
    d: usize,
    seed: Option<u64>,
    iters_mul: Option<f64>,
) -> PyResult<&'py PyArray2<u32>> {
    if d >= n {
        return Err(PyValueError::new_err("require d < n"));
    }
    if (n * d) % 2 != 0 {
        return Err(PyValueError::new_err("n*d must be even"));
    }

    let seed_v = seed.unwrap_or(0xC0FFEEu64);
    let c = iters_mul.unwrap_or(5.0);

    // Compute without holding the GIL.
    let edges = py.allow_threads(|| generate_one(n, d, seed_v, c));

    // Convert to (m,2) u32 array
    let mut flat: Vec<u32> = Vec::with_capacity(edges.len() * 2);
    for (u, v) in edges {
        flat.push(u as u32);
        flat.push(v as u32);
    }

    let arr = PyArray2::<u32>::from_vec2(
        py,
        &flat
            .chunks(2)
            .map(|p| vec![p[0], p[1]])
            .collect::<Vec<_>>(),
    )
    .map_err(|_| PyValueError::new_err("failed to build numpy array"))?;
    Ok(arr)
}

#[pyfunction]
fn d_regular_near_uniform_batch<'py>(
    py: Python<'py>,
    n: usize,
    d: usize,
    seeds: Vec<u64>,
    iters_mul: Option<f64>,
) -> PyResult<&'py PyList> {
    if d >= n {
        return Err(PyValueError::new_err("require d < n"));
    }
    if (n * d) % 2 != 0 {
        return Err(PyValueError::new_err("n*d must be even"));
    }
    if seeds.is_empty() {
        return Err(PyValueError::new_err("seeds must be non-empty"));
    }

    let c = iters_mul.unwrap_or(5.0);

    // Compute all graphs in parallel without GIL (Rayon).
    let all_edges: Vec<Vec<(usize, usize)>> = py.allow_threads(|| {
        seeds
            .par_iter()
            .map(|&s| generate_one(n, d, s, c))
            .collect()
    });

    // Build Python list of numpy arrays.
    let out = PyList::empty(py);
    for edges in all_edges {
        let mut flat: Vec<u32> = Vec::with_capacity(edges.len() * 2);
        for (u, v) in edges {
            flat.push(u as u32);
            flat.push(v as u32);
        }
        let arr = PyArray2::<u32>::from_vec2(
            py,
            &flat
                .chunks(2)
                .map(|p| vec![p[0], p[1]])
                .collect::<Vec<_>>(),
        )
        .map_err(|_| PyValueError::new_err("failed to build numpy array"))?;
        out.append(arr)?;
    }
    Ok(out)
}

#[pymodule]
fn rand_d_regular(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(d_regular_near_uniform, m)?)?;
    m.add_function(wrap_pyfunction!(d_regular_near_uniform_batch, m)?)?;
    Ok(())
}
