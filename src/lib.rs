use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use pyo3::prelude::*;

#[derive(Clone)]
pub struct FastGraph {
    n:  usize,
    d:  usize,
    adj: Vec<HashSet<usize>>,
}

impl FastGraph {
    pub fn new_k_regular(n: usize, d: usize) -> Self {
        assert!(n > 1,               "n ≥ 2");
        assert!(d < n,               "d must be < n");
        assert!((n * d) % 2 == 0,    "n * d must be even");
        assert!(d % 2 == 0 || n % 2 == 0,
                "n must be even when d is odd");

        let mut adj = vec![HashSet::with_capacity(d); n];

        // 1. Add ±1, ±2, … ±⌊d/2⌋ steps (a circulant graph)
        for i in 0..n {
            for k in 1..=d / 2 {
                let next = (i + k) % n;
                let prev = (i + n - k) % n;

                adj[i].insert(next);
                adj[next].insert(i);

                adj[i].insert(prev);
                adj[prev].insert(i);
            }
        }

        // 2. Odd‑degree patch: connect opposite vertices (perfect matching)
        if d % 2 == 1 {
            let half = n / 2;
            for i in 0..half {
                adj[i].insert(i + half);
                adj[i + half].insert(i);
            }
        }

        Self { n, d, adj }
    }

    /// Shuffle edges with double‑edge swaps (mixing_factor * n * d attempts)
    pub fn uniform_mixing(&mut self,
                          rng: &mut impl Rng,
                          mixing_factor: usize)
    {
        if mixing_factor == 0 { return; }

        let attempts = mixing_factor
            .saturating_mul(self.n)
            .saturating_mul(self.d); // protected from usize overflow

        let mut edges = Vec::with_capacity(self.n * self.d / 2);

        for step in 0..attempts {
            // Rebuild edge list every 256 swaps or when requested
            if step % 256 == 0 || edges.is_empty() {
                edges.clear();
                for u in 0..self.n {
                    for &v in &self.adj[u] {
                        if u < v { edges.push((u, v)); }
                    }
                }
                if edges.len() < 2 { break; }
            }

            // Sample two distinct edges
            let (idx1, idx2) = {
                let mut i = rng.gen_range(0..edges.len());
                let mut j = rng.gen_range(0..edges.len());
                while i == j { j = rng.gen_range(0..edges.len()); }
                (i, j)
            };

            let (a, b) = edges[idx1];
            let (c, d) = edges[idx2];

            // Disallow shared vertices
            if a == c || a == d || b == c || b == d { continue; }

            // Decide swap orientation
            let (u1, v1, u2, v2) = if rng.gen_bool(0.5) {
                (a, c, b, d)
            } else {
                (a, d, b, c)
            };

            // Avoid loops / parallel edges
            if u1 == v1
                || u2 == v2
                || self.adj[u1].contains(&v1)
                || self.adj[u2].contains(&v2)
                || (u1 == u2 && v1 == v2)
            {
                continue;
            }

            // Do swap
            self.adj[a].remove(&b);
            self.adj[b].remove(&a);
            self.adj[c].remove(&d);
            self.adj[d].remove(&c);

            self.adj[u1].insert(v1);
            self.adj[v1].insert(u1);
            self.adj[u2].insert(v2);
            self.adj[v2].insert(u2);

            // Update local edge cache
            edges[idx1] = if u1 < v1 { (u1, v1) } else { (v1, u1) };
            edges[idx2] = if u2 < v2 { (u2, v2) } else { (v2, u2) };
        }
    }

    /// Each edge exactly once (`u < v`).
    pub fn to_edge_list(&self) -> Vec<(usize, usize)> {
        let mut edges = Vec::with_capacity(self.n * self.d / 2);
        for (u, nbrs) in self.adj.iter().enumerate() {
            for &v in nbrs {
                if u < v { edges.push((u, v)); }
            }
        }
        edges
    }

    /// Symmetric COO (both directions emitted).
    pub fn to_sparse_arrays(&self)
        -> (Vec<usize>, Vec<usize>, Vec<u8>)
    {
        let mut rows = Vec::with_capacity(self.n * self.d);
        let mut cols = Vec::with_capacity(self.n * self.d);
        let mut data = Vec::with_capacity(self.n * self.d);

        for (u, nbrs) in self.adj.iter().enumerate() {
            for &v in nbrs {
                rows.push(u);
                cols.push(v);
                data.push(1);
            }
        }
        (rows, cols, data)
    }
}

// ---------- Python bindings ----------
#[pyfunction]
fn generate_uniform_regular_sparse(
    n: usize,
    d: usize,
    seed: u64,
    mixing_factor: Option<usize>,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<u8>)> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g   = FastGraph::new_k_regular(n, d);
    g.uniform_mixing(&mut rng, mixing_factor.unwrap_or(5));
    Ok(g.to_sparse_arrays())
}

#[pyfunction]
fn generate_uniform_regular_edges(
    n: usize,
    d: usize,
    seed: u64,
    mixing_factor: Option<usize>,
) -> PyResult<Vec<(usize, usize)>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut g   = FastGraph::new_k_regular(n, d);
    g.uniform_mixing(&mut rng, mixing_factor.unwrap_or(5));
    Ok(g.to_edge_list())
}

#[pyfunction]
fn generate_multiple_graphs_sparse(
    n: usize,
    d: usize,
    count: usize,
    base_seed: u64,
    mixing_factor: Option<usize>,
) -> PyResult<Vec<(Vec<usize>, Vec<usize>, Vec<u8>)>> {
    let mixing = mixing_factor.unwrap_or(5);
    let graphs: Vec<_> = (0..count)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(i as u64));
            let mut g   = FastGraph::new_k_regular(n, d);
            g.uniform_mixing(&mut rng, mixing);
            g.to_sparse_arrays()
        })
        .collect();
    Ok(graphs)
}

#[pyfunction]
fn generate_uniform_regular(  // legacy alias
    n: usize,
    d: usize,
    seed: u64,
    mixing_factor: Option<usize>,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<u8>)> {
    generate_uniform_regular_sparse(n, d, seed, mixing_factor)
}

#[pymodule]
fn fast_graph_gen(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(generate_uniform_regular, m)?)?;
    m.add_function(wrap_pyfunction!(generate_uniform_regular_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(generate_uniform_regular_edges, m)?)?;
    m.add_function(wrap_pyfunction!(generate_multiple_graphs_sparse, m)?)?;
    Ok(())
}