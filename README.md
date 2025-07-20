# Berry–Esseen Universality in Sparse d‑Regular Graphs

This repo **tests the sharp Berry–Esseen bounds** recently proved by  
*Leonhard Nagel (2025)* and *Huang & Yau (2023)* for eigenvector overlaps in sparse random‑regular graphs.

We:

* generate batches of random *d*-regular graphs,
* compute the Kolmogorov–Smirnov distance **D_N** of rescaled eigenvector entries,
* check that **log D_N vs log N** follows the predicted slope **‑1/6** (up to log factors).

---

## Roadmap

| Step | Task | What we expect | Status |
|------|------|----------------|--------|
| **1** | **Fixed degree** | For several (d, N) pairs, slope of log D_N vs log N ≈ ‑1/6 | **Done ✅** |
| **2** | **Growing degree** | For d(N) ≤ N^0.25, slope stays ≈ ‑1/6 after dividing D_N by √d | In progress 🕒 |

---

## Quick start

```bash
# clone repository
git clone https://github.com/talkierbox/berry-essen-universality-bounds
cd berry-essen-universality-bounds

# install dependencies
pip install -r requirements.txt

# generate graph data (writes compressed edge lists to ./data)
python generate_graphs.py

# run the analysis notebook
jupyterlab main.ipynb
```

## Repository Layout
```
.
├── generate_graphs.py   # batch generator using NetworKit / NetworkX
├── main.ipynb           # main notebook
└── data/                # .npz files created by generate_graphs.py
```

## Requirements
Python 3.10+, `pip install -r requirements.txt`