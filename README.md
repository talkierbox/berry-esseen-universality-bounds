# Berry–Esseen Universality in Sparse Random‑Regular Graphs

This repo is an **empirical check** of the Berry–Esseen bounds proved by  
*Leonhard Nagel (2025) and Huang & Yau (2023)* for eigenvector overlaps in sparse random‑regular graphs.
---

## Project roadmap

| # | Goal | Status |
|---|------|--------|
| **1** | **Fixed-degree test**: simulate multiple (d, N) pairs, compute KS distance D_N, confirm the log D_N vs log N slope ≈ -1/6. | **Done ✅** |
| **2** | Growing-degree test: let d(N) ≲ N^0.25, verify D_N ~ sqrt(d) · N^(-1/6) after rescaling. | Planned 🕒 |

---

## Quick start

```bash
# clone & set up
git clone https://github.com/talkierbox/berry-essen-universality-bounds
pip install -r requirements.txt
python generate_graphs.py
jupyterlab main.ipynb
```