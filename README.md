# Shortest Paths on Dynamic Graphs (GPU)

Course project for GPU Programming. We implement **single-source shortest paths (SSSP)** and
**all-pairs shortest paths (APSP)** on **dynamic graphs**, meaning graphs that change over time
through batches of edge insertions, deletions and weight changes. We compare two GPU strategies:

1. **Static recomputation**: after every batch of updates, throw away the old result and rerun SSSP/APSP from scratch.
2. **Dynamic (incremental) update**: after every batch, repair only the parts of the shortest-path tree that the batch affected.

CPU versions serve as a **correctness oracle** and as a **performance baseline**.

---

## Problem definition

- **Input:** a directed, weighted graph `G = (V, E, w)` with non-negative weights, a source `s` (SSSP),
  and a stream of update batches `B_1, B_2, ..., B_k`.
- **Update batch:** a list of operations
  - `INSERT(u, v, w)`: add an edge
  - `DELETE(u, v)`: remove an edge
  - `UPDATE(u, v, w')`: change a weight (handled as delete + insert, or directly)
- **Output after each batch:**
  - SSSP: `dist[v]` and `parent[v]` for every vertex `v`
  - APSP: the distance matrix `D[u][v]` (only on smaller graphs, since it needs O(V²) memory)

## Algorithms

### Static SSSP (recompute per batch)
| Target | Algorithm | Notes |
|--------|-----------|-------|
| CPU | Dijkstra (binary heap) | Reference output used to check every other implementation |
| GPU | Bellman-Ford (frontier-based) | Simple baseline: one thread per vertex or edge, relax until nothing changes |
| GPU | Delta-stepping | Faster on large graphs; uses buckets of width Δ |

### Dynamic SSSP (batch update)
Based on the two-phase approach used in most parallel dynamic SSSP work
(e.g. Khanda et al., *"A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks"*, IEEE TPDS 2022):

1. **Identify affected vertices**
   - *Insertion* `(u, v, w)`: if `dist[u] + w < dist[v]`, then `v` is affected and its distance goes down.
   - *Deletion* `(u, v)`: if `parent[v] == u`, then `v` and its whole subtree in the SSSP tree are affected, so their distances are set to ∞.
2. **Propagate**: relax edges starting only from the affected vertices, repeating until nothing changes.
   Deletions are done first (marking subtrees invalid), then insertions plus the repair pass.

The point of the project is to measure **when the dynamic approach wins over recomputing from scratch**,
for example as batch size grows or as the insert/delete mix changes.

### APSP
- **Static:** blocked Floyd-Warshall on the GPU (dense, small/medium graphs), or V runs of SSSP for sparse graphs.
- **Dynamic:** run dynamic SSSP from every source (or from a sampled subset of sources). The batch of updates is shared, so the same affected-edge set is reused for every source.

## Graph storage

The same algorithms run on top of different graph representations. These live behind one common API:

| Format | Pros | Cons |
|--------|------|------|
| **CSR** (compressed sparse row) | Coalesced reads, the standard format for static GPU graphs | Every insertion/deletion means rebuilding the arrays |
| **COO / edge list** | Updates are trivial; works well for edge-parallel kernels | Poor locality for vertex-centric traversal |
| **CSR with slack / per-vertex blocks** (Hornet / faimGraph style) | Fast in-place updates on the GPU | More complex memory management |
| **Adjacency list** (CPU) | Simple, flexible | CPU only |

## Proposed repository layout

```
.
├── include/
│   ├── graph/          # Graph API: CSR, COO, dynamic CSR (CPU + GPU)
│   └── common/         # update batch format, timers, CUDA error checks
├── src/
│   ├── graph/          # graph storage implementations
│   ├── cpu/            # Dijkstra, CPU dynamic SSSP, CPU APSP
│   ├── gpu/            # Bellman-Ford, delta-stepping, dynamic SSSP, APSP kernels
│   └── main.cu         # driver: load graph, apply batches, run selected algorithm
├── tools/
│   ├── gen_updates.py  # generates random update batches
│   └── plot.py         # turns benchmark CSVs into plots
├── tests/              # compares every implementation against CPU Dijkstra
├── data/               # datasets (gitignored, download script only)
├── results/            # benchmark CSVs and plots
├── README.md
└── division.md         # who does what
```

## Building and running (planned)

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j

# run dynamic SSSP on the GPU with CSR storage, 10 batches of 10k updates each
./dyn_sssp --graph ../data/roadNet-CA.mtx --format csr --algo gpu-dynamic \
           --source 0 --batches 10 --batch-size 10000 --insert-ratio 0.5

# check against CPU Dijkstra
./dyn_sssp ... --verify
```

## Datasets

- **Road networks** (high diameter, low degree): DIMACS 9th Challenge (USA-road-d.*), SNAP `roadNet-CA`
- **Social / web graphs** (low diameter, skewed degree): SNAP `soc-LiveJournal1`, `web-Google`
- **Synthetic:** R-MAT / Erdős–Rényi graphs generated at different sizes
- Unweighted graphs get random integer weights in `[1, 100]`.

## Evaluation plan

For every (graph, storage format, algorithm) combination we measure:
- **Time per batch** (split into update-apply time and SSSP time)
- **Speedup** of GPU dynamic over GPU static, and of GPU over CPU
- How the speedup changes with **batch size** (100 up to 1M updates) and **insert/delete ratio**
- **Affected-vertex count** per batch, to explain why dynamic wins or loses
- **Correctness:** every GPU result is checked against CPU Dijkstra (`dist` must match exactly)

## Team and work division

Summary below. Full details, interfaces and milestones are in [`division.md`](division.md).

| Member | Area | Main deliverables |
|--------|------|-------------------|
| **Person 1** | GPU static | GPU Bellman-Ford + delta-stepping, recomputed after each batch; static GPU APSP (Floyd-Warshall) |
| **Person 2** | GPU dynamic | GPU batch-dynamic SSSP (affected-vertex marking, subtree invalidation, repair); dynamic APSP built on top of it |
| **Person 3** | CPU reference | CPU Dijkstra (correctness oracle), CPU dynamic SSSP (to check and compare against the GPU dynamic version), test suite |
| **Person 4** | Graph API + infra | CPU/GPU graph storage (CSR, COO, dynamic CSR), update-batch generator, data loaders, benchmark harness and plots |

## References

1. A. Khanda, S. Srinivasan, S. Bhowmick, B. Norris, S. K. Das. *A Parallel Algorithm Template for Updating Single-Source Shortest Paths in Large-Scale Dynamic Networks.* IEEE TPDS, 2022.
2. U. Meyer, P. Sanders. *Δ-stepping: a parallelizable shortest path algorithm.* J. Algorithms, 2003.
3. F. Busato, O. Green, N. Bombieri, D. Bader. *Hornet: An Efficient Data Structure for Dynamic Sparse Graphs and Matrices on GPUs.* HPEC, 2018.
4. M. Winter et al. *faimGraph: High Performance Management of Fully-Dynamic Graphs under tight Memory Constraints on the GPU.* SC, 2018.
5. G. Ramalingam, T. Reps. *On the computational complexity of dynamic graph problems.* Theoretical Computer Science, 1996.
