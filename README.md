# Berry–Esseen Universality in Sparse d‑Regular Graphs

This repo **tests the sharp Berry–Esseen bounds** recently proved by  
*Leonhard Nagel (2025)* and *Huang & Yau (2023)* for eigenvector overlaps in sparse random‑regular graphs.

We:

* generate batches of random *d*-regular graphs,
* compute the Kolmogorov–Smirnov distance **D_N** of rescaled eigenvector entries,
* check that **log D_N vs log N** follows the predicted slope **‑1/6** (up to log factors).

---

## Performance Modes

This project supports two implementations for maximum performance:

🦀 **Rust Mode** (Recommended): Ultra-fast graph generation using Rust + PyO3  
🐍 **Python Mode**: Pure Python/Numba fallback (no Rust installation required)

---

## Installation

### Option 1: Full Installation (Rust + Python) - Recommended

```bash
# Clone repository
git clone https://github.com/talkierbox/berry-essen-universality-bounds
cd berry-essen-universality-bounds

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install maturin for building Rust extensions
pip install maturin

# Build and install the Rust extension
maturin develop --release
```

### Option 2: Python-Only Installation

```bash
# Clone repository
git clone https://github.com/talkierbox/berry-essen-universality-bounds
cd berry-essen-universality-bounds

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

---

## Usage

### Generate Graph Data

```bash
# Generate graphs (automatically uses Rust if available)
python generate_graphs.py
```

### Run Analysis

```bash
# Run the analysis notebook
jupyter lab main.ipynb
```

---

## Roadmap

| Step | Task | What we expect | Status |
|------|------|----------------|--------|
| **1** | **Fixed degree** | For several (d, N) pairs, slope of log D_N vs log N ≈ ‑1/6 | **Done ✅** |
| **2** | **Growing degree** | For d(N) ≤ N^0.25, slope stays ≈ ‑1/6 after dividing D_N by √d |  **Done ✅** |

---

## Repository Layout

```
.
├── generate_graphs.py   # hybrid generator (Rust/Python modes)
├── src/
│   └── lib.rs          # fast Rust implementation
├── Cargo.toml          # Rust dependencies
├── main.ipynb          # main analysis notebook
└── data/               # .npz files created by generate_graphs.py
```

## Dependencies

- **Python 3.10+** with packages in `requirements.txt`
- **Rust 1.70+** (optional, for performance mode)
- **maturin** (for building Rust extensions)
