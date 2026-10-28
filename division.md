# Work Division

Four members, four areas. Everyone codes against the **shared interfaces** defined below, which lets
the work go on in parallel from the first week.

---

## Shared interfaces (agree on these in week 1)

Person 4 owns these headers, but everyone should review them before they are frozen.

```cpp
// Edge update in a batch
enum class UpdateType : uint8_t { INSERT, DELETE, UPDATE };
struct EdgeUpdate { int u, v; float w; UpdateType type; };
struct UpdateBatch { std::vector<EdgeUpdate> updates; };   // + device copy

// Result every SSSP implementation must produce
struct SSSPResult {
    std::vector<float> dist;    // INF if unreachable
    std::vector<int>   parent;  // -1 for the source / unreachable vertices
};

// Graph API (implemented separately for CPU and GPU, per format)
class Graph {
    int  num_vertices() const;
    int  num_edges() const;
    void apply_batch(const UpdateBatch&);   // mutate the graph
    // + format-specific raw accessors: row_ptr / col_idx / weights (CSR), etc.
};

// Common algorithm signature
SSSPResult run_sssp(const Graph&, int source);
void       update_sssp(Graph&, const UpdateBatch&, SSSPResult& in_out);  // dynamic versions only
```

**Rule:** until Person 4 delivers the real graph classes, everyone works on a plain CSR built from an
edge list, with the same field names, so switching over later is a small change.

---

## Person 1: GPU static SSSP (recompute per batch)

**Goal:** the GPU baseline. After every batch, rebuild the graph and compute SSSP from scratch.

Tasks:
1. **GPU Bellman-Ford (frontier-based):** vertex-parallel relaxation with `atomicMin` on distances,
   plus a frontier queue so each iteration only touches vertices that changed.
2. **GPU delta-stepping:** bucketed version, with Δ tuned per graph type (road vs. social).
3. **Recompute loop:** for each batch, call Person 4's `apply_batch`, run SSSP, and record
   update time and compute time separately.
4. **Static APSP on the GPU:** blocked Floyd-Warshall in shared memory for small and dense graphs,
   and multi-source SSSP (batched sources) for sparse graphs.
5. Optimizations to try and report on: warp-per-vertex for high-degree vertices, coalesced CSR reads,
   reducing atomic contention.

Deliverables: `src/gpu/static_sssp.cu`, `src/gpu/static_apsp.cu`, timings per batch.

---

## Person 2: GPU dynamic SSSP (batch updates)

**Goal:** the main contribution. Update `dist`/`parent` after a batch without recomputing everything.

Tasks:
1. **Process deletions:** for each deleted edge `(u,v)` with `parent[v]==u`, mark `v` as affected and
   set `dist[v]=INF`. Then **propagate the invalidation down the SSSP tree** (iteratively, in parallel)
   to mark the whole subtree.
2. **Process insertions:** for each inserted edge `(u,v,w)` with `dist[u]+w < dist[v]`, update `v` and mark it affected.
3. **Repair phase:** relax edges outward from the affected set until nothing changes. Invalidated
   vertices must also pull distances from their *incoming* neighbours, so this needs either reverse
   CSR or a pull-style kernel. Coordinate this with Person 4.
4. **Batch handling:** deduplicate conflicting updates inside a batch (e.g. insert and delete of the same edge).
5. **Dynamic APSP:** run the dynamic update for many sources at once (one source per block, or batched), reusing the batch.
6. Experiment: find the **crossover point** where recomputing from scratch (Person 1) becomes faster than the dynamic update.

Deliverables: `src/gpu/dynamic_sssp.cu`, `src/gpu/dynamic_apsp.cu`, crossover analysis.

> This is the hardest role. Persons 1 and 3 should pair with Person 2 on debugging, especially on deletions.

---

## Person 3: CPU reference + CPU dynamic SSSP

**Goal:** make sure every result is correct, and give the CPU baselines to compare against.

Tasks:
1. **CPU Dijkstra** (binary heap): this is the ground truth for everything else.
2. **CPU static recompute per batch:** baseline timing for Person 1's GPU version.
3. **CPU dynamic SSSP:** the same algorithm as Person 2's, written sequentially (and optionally
   with OpenMP). Since it uses the same steps, it can be used to debug the GPU version one step at a
   time by comparing the affected sets and intermediate distances.
4. **Test suite:**
   - Small hand-made graphs with known answers (e.g. a deletion that disconnects a vertex,
     a deletion with an equal-cost alternative path, a zero-weight edge)
   - Randomized tests: random graph + random batches, compared against Dijkstra after **every** batch
   - Checks that `dist` matches exactly and that `parent` gives a valid shortest-path tree
     (parents may differ when two paths have equal cost, which is fine)
5. CPU APSP (V runs of Dijkstra) to validate APSP on small graphs.

Deliverables: `src/cpu/*`, `tests/*`, a `--verify` flag in the driver.

---

## Person 4: Graph storage API (CPU + GPU) + infrastructure

**Goal:** the shared base layer. This work blocks everyone else, so the basic version has to be done early.

Tasks:
1. **Define and freeze the shared headers** (above) in week 1.
2. **Graph formats:**
   - CPU: adjacency list, CSR
   - GPU: CSR (rebuilt per batch), COO / edge list, and one **GPU-updatable format**
     (CSR with slack space per vertex, or Hornet-style blocks) so updates don't need a full rebuild
   - Reverse CSR (incoming edges), which Person 2 needs for the repair phase
3. **`apply_batch` on GPU:** parallel insert/delete in each format; measure its cost on its own.
4. **I/O:** loaders for Matrix Market / SNAP / DIMACS formats, plus random weight assignment.
5. **Update batch generator:** configurable size and insert/delete ratio, guaranteed to only delete edges that exist.
6. **Benchmark harness + plots:** a driver that runs any (graph, format, algorithm) combination and writes CSV, then `plot.py`.
7. **Build system:** CMake, CUDA error-check macros, timers.

Deliverables: `include/graph/*`, `src/graph/*`, `tools/*`, `CMakeLists.txt`, `src/main.cu`.

---

## Timeline (suggested, 6 weeks)

| Week | P1 (GPU static) | P2 (GPU dynamic) | P3 (CPU) | P4 (API/infra) |
|------|-----------------|------------------|----------|----------------|
| 1 | Read up, simple BF kernel on plain CSR | Read Khanda et al., design kernels | Dijkstra + small test graphs | **Freeze headers**, CMake, loaders |
| 2 | Frontier BF working | Insertion-only dynamic | Randomized test harness | CPU/GPU CSR + batch generator |
| 3 | Delta-stepping | Deletions + subtree invalidation | CPU dynamic SSSP | Reverse CSR, COO, GPU `apply_batch` |
| 4 | Static APSP (Floyd-Warshall) | Full mixed batches, verify vs P3 | Help P2 debug, CPU APSP | Updatable GPU format (slack CSR) |
| 5 | Optimize + run on all formats | Dynamic APSP, crossover study | Verify all combos | Benchmark harness, plots |
| 6 | **Everyone:** final benchmarks, report, slides | | | |

---

## Is this split fair for 4 people?

Mostly yes. Each person owns a clear layer and the interfaces between layers are small. Things to watch out for:

- **APSP was not assigned to anyone in the original split.** It is now given to Person 1 (static)
  and Person 2 (dynamic), because it builds directly on their SSSP code.
- **Person 2 has the hardest job.** Dynamic deletions are hard to get right. Person 3's CPU dynamic
  version follows the same steps on purpose, so the two of them can debug together.
- **Person 3's static part (Dijkstra) is small**, so they also own the test suite and the CPU dynamic
  version. That keeps their total workload close to the others'.
- **Person 4 blocks everyone else.** The headers and a basic CSR need to be done in week 1. After that,
  the updatable GPU format and the benchmark harness are a full role of their own.
- **Everyone** is responsible for timing their own code with the shared harness, and for writing their own section of the final report.
