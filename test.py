#!/usr/bin/env python3
"""
Test script for d-regular graph generation.
Verifies that the Rust backend correctly generates uniform d-regular graphs.
"""

import numpy as np
import scipy.sparse as sp
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Set, Tuple
from tqdm import tqdm
import sys

def test_rust_backend():
    """Test if Rust backend is available and working."""
    try:
        from rand_d_regular import d_regular_near_uniform
        return True
    except ImportError:
        print("❌ Rust backend not available")
        return False

def to_csr(n: int, edges: np.ndarray) -> sp.csr_matrix:
    """Convert edge array to CSR matrix."""
    # edges is (m,2) array with undirected edges
    edges = np.asarray(edges, dtype=np.int32)
    u = edges[:, 0]
    v = edges[:, 1]
    m = len(edges)
    
    # Create symmetric matrix
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    data = np.ones(2 * m, dtype=np.uint8)
    
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.uint8)

def get_edge_set(A: sp.csr_matrix) -> Set[Tuple[int, int]]:
    """Convert sparse matrix to set of edges (undirected)."""
    coo = A.tocoo()
    edges = set()
    for i, j in zip(coo.row, coo.col):
        if i < j:  # Only store each edge once (undirected)
            edges.add((i, j))
    return edges

def test_d_regularity(n: int, d: int, num_tests: int = 10):
    """Test that generated graphs are d-regular."""
    print(f"🔍 Testing d-regularity for n={n}, d={d}")
    
    if not test_rust_backend():
        return False
    
    from rand_d_regular import d_regular_near_uniform
    
    issues = []
    
    with tqdm(range(num_tests), desc="Checking d-regularity", leave=False) as pbar:
        for seed in pbar:
            edges = d_regular_near_uniform(n, d, seed, 5.0)
            A = to_csr(n, edges)
            
            # Check degree sequence
            degrees = np.array(A.sum(axis=1)).flatten()
            
            if not np.all(degrees == d):
                issues.append(f"Seed {seed}: Not d-regular! Degrees: {np.unique(degrees, return_counts=True)}")
            
            # Check expected number of edges
            expected_edges = n * d // 2
            actual_edges = A.nnz // 2
            if actual_edges != expected_edges:
                issues.append(f"Seed {seed}: Wrong edge count! Expected: {expected_edges}, Got: {actual_edges}")
            
            # Check symmetry
            if (A != A.T).nnz > 0:
                issues.append(f"Seed {seed}: Graph not symmetric!")
    
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   {issue}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")
        return False
    else:
        print(f"✅ All {num_tests} graphs are d-regular")
        return True

def test_simple_graph(n: int, d: int, num_tests: int = 10):
    """Test that generated graphs are simple (no self-loops, no multi-edges)."""
    print(f"🔍 Testing simplicity for n={n}, d={d}")
    
    if not test_rust_backend():
        return False
    
    from rand_d_regular import d_regular_near_uniform
    
    issues = []
    
    with tqdm(range(num_tests), desc="Checking simplicity", leave=False) as pbar:
        for seed in pbar:
            edges = d_regular_near_uniform(n, d, seed, 5.0)
            A = to_csr(n, edges)
            
            # Check for self-loops
            if A.diagonal().sum() > 0:
                issues.append(f"Seed {seed}: Has self-loops!")
            
            # Check for multi-edges (all data should be 1)
            if not np.all(A.data == 1):
                issues.append(f"Seed {seed}: Has multi-edges! Data values: {np.unique(A.data)}")
    
    if issues:
        print(f"❌ Found {len(issues)} issues:")
        for issue in issues[:5]:
            print(f"   {issue}")
        if len(issues) > 5:
            print(f"   ... and {len(issues) - 5} more")
        return False
    else:
        print(f"✅ All {num_tests} graphs are simple")
        return True

def test_uniformity(n: int, d: int, num_tests: int = 100):
    """Test uniformity by checking that different graphs are generated."""
    print(f"🔍 Testing uniformity for n={n}, d={d}")
    
    if not test_rust_backend():
        return False
    
    from rand_d_regular import d_regular_near_uniform
    
    # Generate multiple graphs and collect them
    graphs = []
    
    with tqdm(range(num_tests), desc="Generating graphs", leave=False) as pbar:
        for seed in pbar:
            edges = d_regular_near_uniform(n, d, seed, 5.0)
            A = to_csr(n, edges)
            edge_set = get_edge_set(A)
            graphs.append(frozenset(edge_set))
    
    # Check uniqueness
    unique_graphs = len(set(graphs))
    print(f"   Generated {unique_graphs}/{num_tests} unique graphs ({unique_graphs/num_tests*100:.1f}%)")
    
    # For small graphs, we expect high uniqueness
    if n <= 20:
        min_unique_rate = 0.8  # 80%
    else:
        min_unique_rate = 0.5  # 50% for larger graphs
    
    if unique_graphs / num_tests >= min_unique_rate:
        print(f"✅ Good uniqueness rate (≥{min_unique_rate*100:.0f}%)")
        return True
    else:
        print(f"❌ Low uniqueness rate (<{min_unique_rate*100:.0f}%)")
        return False

def test_edge_distribution(n: int, d: int, num_tests: int = 1000):
    """Test that edge distribution looks uniform."""
    print(f"🔍 Testing edge distribution for n={n}, d={d}")
    
    if not test_rust_backend():
        return False
    
    from rand_d_regular import d_regular_near_uniform
    
    # Count how often each edge appears
    edge_counts = defaultdict(int)
    
    with tqdm(range(num_tests), desc="Analyzing edge distribution", leave=False) as pbar:
        for seed in pbar:
            edges = d_regular_near_uniform(n, d, seed, 5.0)
            A = to_csr(n, edges)
            edges_set = get_edge_set(A)
            
            for edge in edges_set:
                edge_counts[edge] += 1
    
    # Analyze distribution
    counts = list(edge_counts.values())
    if counts:
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        min_count = np.min(counts)
        max_count = np.max(counts)
        cv = std_count / mean_count if mean_count > 0 else float('inf')
        
        print(f"   Edge statistics: μ={mean_count:.2f}, σ={std_count:.2f}, CV={cv:.3f}")
        print(f"   Range: [{min_count}, {max_count}]")
        
        # For uniform distribution, we expect relatively low coefficient of variation
        if cv < 0.55:  # Reasonable threshold
            print(f"✅ Edge distribution looks reasonably uniform")
            return True
        else:
            print(f"❌ Edge distribution may not be uniform (high CV)")
            return False
    else:
        print(f"❌ No edges found!")
        return False

def benchmark_speed(n: int, d: int, num_graphs: int = 100):
    """Benchmark graph generation speed."""
    print(f"⏱️  Benchmarking speed for n={n}, d={d}")
    
    if not test_rust_backend():
        return
    
    from rand_d_regular import d_regular_near_uniform, d_regular_near_uniform_batch
    import time
    
    # Single graph generation
    start = time.time()
    with tqdm(range(num_graphs), desc="Single generation", leave=False) as pbar:
        for i in pbar:
            edges = d_regular_near_uniform(n, d, i, 5.0)
    single_time = time.time() - start
    
    # Batch generation if available
    try:
        start = time.time()
        print("   Running batch generation...", end="", flush=True)
        seeds = list(range(num_graphs))
        batch_result = d_regular_near_uniform_batch(n, d, seeds, 5.0)
        batch_time = time.time() - start
        print(" Done!")
        
        print(f"   Single: {single_time:.3f}s ({single_time/num_graphs*1000:.2f}ms/graph)")
        print(f"   Batch:  {batch_time:.3f}s ({batch_time/num_graphs*1000:.2f}ms/graph)")
        print(f"   Speedup: {single_time/batch_time:.2f}x")
    except (ImportError, AttributeError):
        print("   Batch generation not available")
        print(f"   Single: {single_time:.3f}s ({single_time/num_graphs*1000:.2f}ms/graph)")

def benchmark_rust_vs_python(n: int, d: int, num_graphs: int = 20):
    """Compare Rust backend speed vs Python/Numba implementation."""
    print(f"⚡ Rust vs Python speed comparison for n={n}, d={d}")
    
    # Skip if graph is too large for Python backend
    if n > 10000:
        print("   ⚠️  Skipping Python benchmark - graph too large")
        return
    
    import time
    
    # Test Python backend (import from generate_graphs.py)
    try:
        from generate_graphs import _python_graph
        python_available = True
    except ImportError:
        print("   ❌ Could not import Python backend")
        return
    
    # Test Rust backend
    if not test_rust_backend():
        print("   ❌ Rust backend not available")
        return
    
    from rand_d_regular import d_regular_near_uniform
    
    print(f"   Testing with {num_graphs} graphs...")
    
    # Benchmark Python implementation
    print("   Running Python backend...", end="", flush=True)
    start = time.time()
    python_graphs = []
    for i in range(num_graphs):
        A = _python_graph(n, d, i, 3)
        python_graphs.append(A)
    python_time = time.time() - start
    print(" Done!")
    
    # Benchmark Rust implementation  
    print("   Running Rust backend...", end="", flush=True)
    start = time.time()
    rust_graphs = []
    for i in range(num_graphs):
        edges = d_regular_near_uniform(n, d, i, 5.0)
        A = to_csr(n, edges)
        rust_graphs.append(A)
    rust_time = time.time() - start
    print(" Done!")
    
    # Results
    python_rate = python_time / num_graphs * 1000
    rust_rate = rust_time / num_graphs * 1000
    speedup = python_time / rust_time if rust_time > 0 else float('inf')
    
    print(f"   📊 Results:")
    print(f"      Python: {python_time:.3f}s ({python_rate:.2f}ms/graph)")
    print(f"      Rust:   {rust_time:.3f}s ({rust_rate:.2f}ms/graph)")
    print(f"      Speedup: {speedup:.1f}x faster with Rust")
    
    # Verify results are equivalent (check first graph)
    if python_graphs and rust_graphs:
        p_edges = get_edge_set(python_graphs[0])
        r_edges = get_edge_set(rust_graphs[0])
        if len(p_edges) == len(r_edges):
            print(f"   ✅ Both backends generate graphs with {len(p_edges)} edges")
        else:
            print(f"   ⚠️  Edge count mismatch: Python={len(p_edges)}, Rust={len(r_edges)}")

def run_test_suite(n: int, d: int):
    """Run all tests for a given configuration."""
    print(f"\n📊 Testing n={n}, d={d}")
    print("─" * 50)
    
    # Check if configuration is valid
    if (n * d) % 2 != 0:
        print(f"❌ Invalid: n*d must be even")
        return False
    
    if d % 2 == 1 and n % 2 != 0:
        print(f"❌ Invalid: n must be even when d is odd")  
        return False
    
    if d >= n:
        print(f"❌ Invalid: d must be < n")
        return False
    
    # Adjust test sizes based on graph size
    if n <= 20:
        uniformity_tests = 50
        edge_dist_tests = 100
    elif n <= 100:
        uniformity_tests = 30
        edge_dist_tests = 50
    else:
        uniformity_tests = 20
        edge_dist_tests = 30
    
    # Run tests
    tests = [
        test_d_regularity(n, d, 10),
        test_simple_graph(n, d, 10),
        test_uniformity(n, d, uniformity_tests),
        test_edge_distribution(n, d, edge_dist_tests),
    ]
    
    passed = sum(tests)
    total = len(tests)
    
    if passed == total:
        print(f"✅ All {total} tests passed")
    else:
        print(f"❌ {passed}/{total} tests passed")
    
    return passed == total

def main():
    """Run all tests."""
    print("🦀 Testing Rust d-regular graph generation")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        (10, 3),   # Small graph
        (20, 3),   # Medium graph  
        (100, 5),  # Larger graph
        (16, 7),   # Odd degree
        (1000, 3), # Large n, small d
        (1000, 5), # Large n, larger d
    ]
    
    # Only test very large graphs if explicitly requested
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        test_configs.extend([
            (10000, 3), # Very large n, small d
            (10000, 5), # Very large n, larger d
        ])
    
    results = []
    
    for n, d in test_configs:
        result = run_test_suite(n, d)
        results.append((n, d, result))
    
    # Speed benchmarks
    print(f"\n📊 Speed benchmarks")
    print("─" * 50) 
    
    # Rust-only benchmark
    print("\n🦀 Rust backend performance:")
    benchmark_speed(100, 3, 50)
    
    # Rust vs Python comparison
    print(f"\n⚔️  Backend comparison:")
    benchmark_rust_vs_python(100, 3, 10)
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        benchmark_rust_vs_python(10000, 5, 10)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("─" * 60)
    
    passed = 0
    for n, d, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   n={n:5d}, d={d}: {status}")
        if result:
            passed += 1
    
    total = len(results)
    print("─" * 60)
    
    if passed == total:
        print(f"🎉 All {total} configurations passed!")
    else:
        print(f"❌ {passed}/{total} configurations passed")
    
    print("\nTip: Use --full flag to test very large graphs (slow)")
    
    return passed == total

if __name__ == "__main__":
    main()
