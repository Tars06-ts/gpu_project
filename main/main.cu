#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#include "common/loader.h"
#include "common/batch_upload.h"
#include "graph/csr.cuh"
#include "graph/tree.cuh"
#include "graph/types.cuh"
#include "graph/checks.cuh"
#include "SSSP/sssp_update.cuh"

namespace fs = std::filesystem;

struct DijkstraResult {
    std::vector<int> dist;
    std::vector<int> parent;
    double time_ms;
};

static DijkstraResult run_cpu_dijkstra(
    int max_n_v,
    int source,
    const std::map<std::pair<int, int>, int>& edges
) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<std::pair<int, int>>> adj(max_n_v);
    for (const auto& [uv, w] : edges) {
        if (uv.first >= 0 && uv.first < max_n_v && uv.second >= 0 && uv.second < max_n_v) {
            adj[uv.first].emplace_back(uv.second, w);
        }
    }

    DijkstraResult res;
    res.dist.assign(max_n_v, INF_DIST);
    res.parent.assign(max_n_v, -1);

    using P = std::pair<int, int>; // (distance, vertex)
    std::priority_queue<P, std::vector<P>, std::greater<P>> pq;

    if (source >= 0 && source < max_n_v) {
        res.dist[source] = 0;
        pq.emplace(0, source);
    }

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d > res.dist[u]) continue;

        for (auto [v, w] : adj[u]) {
            if (res.dist[u] != INF_DIST && res.dist[u] + w < res.dist[v]) {
                res.dist[v] = res.dist[u] + w;
                res.parent[v] = u;
                pq.emplace(res.dist[v], v);
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return res;
}

static int compare_sssp(int max_n_v, Tree& tree, const DijkstraResult& dij, const std::string& label) {
    tree.sync_to_host();
    int mismatches = 0;
    for (int v = 0; v < max_n_v; ++v) {
        int gpu_d = tree.dis(v);
        int cpu_d = dij.dist[v];
        if (gpu_d != cpu_d) {
            if (mismatches < 10) {
                std::cerr << "[" << label << "] MISMATCH at vertex " << v
                          << ": GPU dist=" << gpu_d << " (parent=" << tree.get_parent(v) << ")"
                          << ", Dijkstra dist=" << cpu_d << " (parent=" << dij.parent[v] << ")\n";
            }
            mismatches++;
        }
    }
    return mismatches;
}

static std::string resolve_data_path(const std::string& arg) {
    if (fs::exists(arg)) return arg;
    fs::path in_data = fs::path("data") / arg;
    if (fs::exists(in_data)) return in_data.string();
    return arg;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <graph_file_in_data> <batch_file_in_data> [--source <id>] [--one-indexed] [--max-vertices <N>] [--out-prefix <name>]\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    std::string graph_file = resolve_data_path(argv[1]);
    std::string batch_file = resolve_data_path(argv[2]);
    std::string out_prefix = "sssp";
    int source = 0;
    LoadOptions opt;

    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--source") == 0 && i + 1 < argc) {
            source = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--one-indexed") == 0) {
            opt.one_indexed = true;
        } else if (std::strcmp(argv[i], "--max-vertices") == 0 && i + 1 < argc) {
            opt.max_vertices = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--out-prefix") == 0 && i + 1 < argc) {
            out_prefix = argv[++i];
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        // 1. Load initial graph and batches from data/
        GraphData g_data = load_graph(graph_file, opt);
        if (opt.one_indexed) source -= 1;

        if (source < 0 || source >= g_data.max_n_v) {
            std::cerr << "Error: invalid source vertex " << source << "\n";
            return 1;
        }

        std::vector<HostBatch> host_batches;
        load_batch(batch_file, host_batches, opt);

        // Mirror exact active edge set on CPU for Dijkstra ground-truth verification
        std::map<std::pair<int, int>, int> ref_edges;
        for (const auto& e : g_data.edges) {
            if (e.u >= 0 && e.u < g_data.max_n_v && e.v >= 0 && e.v < g_data.max_n_v) {
                ref_edges[{e.u, e.v}] = e.weight;
            }
        }

        size_t init_n_e = g_data.edges.size();
        std::cout << "Loaded graph: n_v=" << g_data.n_v
                  << ", max_n_v=" << g_data.max_n_v
                  << ", n_e=" << init_n_e
                  << ", batches=" << host_batches.size() << "\n";

        // 2. Initialize empty CSR and Tree on GPU
        CSR graph(g_data.n_v, g_data.max_n_v);
        if (graph.to_gpu() == nullptr) {
            std::cerr << "Error: failed to upload CSR to GPU\n";
            return 1;
        }

        Tree tree(g_data.max_n_v, source);
        tree.to_gpu();

        // 3. Build initial graph G_0 and initial SSSP tree T_0 via update_sssp
        HostBatch init_host_batch;
        init_host_batch.ins = std::move(g_data.edges);
        Batch init_batch = upload_batch(init_host_batch);

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        update_sssp(graph, tree, init_batch.ins, init_batch.n_ins, nullptr, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float init_gpu_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&init_gpu_ms, start, stop));
        free_batch(init_batch);

        // Verify initial T_0 against CPU Dijkstra
        DijkstraResult dij = run_cpu_dijkstra(g_data.max_n_v, source, ref_edges);
        int init_mismatches = compare_sssp(g_data.max_n_v, tree, dij, "init");
        std::cout << "Init T_0: GPU=" << init_gpu_ms << " ms | Dijkstra=" << dij.time_ms
                  << " ms | Mismatches=" << init_mismatches << "\n";

        // 4. Prepare result/ directory and timing log
        fs::create_directories("result");
        std::ofstream timing_out(fs::path("result") / (out_prefix + "_timings.csv"));
        timing_out << "batch,n_ins,n_del,n_rem,gpu_ms,dijkstra_ms,mismatches\n";
        timing_out << "init," << init_n_e << ",0,0," << init_gpu_ms << ","
                   << dij.time_ms << "," << init_mismatches << "\n";

        // 5. Process each dynamic batch on GPU, run Dijkstra on CPU, and verify
        double total_gpu_ms = 0.0;
        double total_cpu_ms = 0.0;
        int total_mismatches = init_mismatches;

        for (size_t b_idx = 0; b_idx < host_batches.size(); ++b_idx) {
            Batch d_batch = upload_batch(host_batches[b_idx]);

            for (int r = 0; r < d_batch.n_rem; ++r) {
                int rem_v = host_batches[b_idx].rem[r];
                graph.remove_vertex(rem_v);
                for (auto it = ref_edges.begin(); it != ref_edges.end(); ) {
                    if (it->first.first == rem_v || it->first.second == rem_v) {
                        it = ref_edges.erase(it);
                    } else {
                        ++it;
                    }
                }
            }

            // Mirror deletions and insertions on CPU reference graph
            for (const auto& e : host_batches[b_idx].del) {
                ref_edges.erase({e.u, e.v});
            }
            for (const auto& e : host_batches[b_idx].ins) {
                if (e.u >= 0 && e.u < g_data.max_n_v && e.v >= 0 && e.v < g_data.max_n_v) {
                    ref_edges[{e.u, e.v}] = e.weight;
                }
            }

            CUDA_CHECK(cudaEventRecord(start));
            update_sssp(graph, tree, d_batch.ins, d_batch.n_ins, d_batch.del, d_batch.n_del);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float gpu_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
            total_gpu_ms += gpu_ms;

            dij = run_cpu_dijkstra(g_data.max_n_v, source, ref_edges);
            total_cpu_ms += dij.time_ms;

            int mismatches = compare_sssp(g_data.max_n_v, tree, dij, "Batch " + std::to_string(b_idx));
            total_mismatches += mismatches;

            std::cout << "Batch " << b_idx
                      << " (ins=" << d_batch.n_ins
                      << ", del=" << d_batch.n_del
                      << ", rem=" << d_batch.n_rem
                      << "): GPU=" << gpu_ms << " ms | Dijkstra=" << dij.time_ms
                      << " ms | Mismatches=" << mismatches << "\n";

            timing_out << b_idx << ","
                       << d_batch.n_ins << ","
                       << d_batch.n_del << ","
                       << d_batch.n_rem << ","
                       << gpu_ms << ","
                       << dij.time_ms << ","
                       << mismatches << "\n";

            free_batch(d_batch);
        }
        timing_out.close();

        // 6. Write final GPU and Dijkstra SSSP outputs to result/
        tree.sync_to_host();
        std::ofstream gpu_out(fs::path("result") / (out_prefix + "_output.txt"));
        std::ofstream dij_out(fs::path("result") / (out_prefix + "_dijkstra.txt"));
        gpu_out << "# vertex parent distance\n";
        dij_out << "# vertex parent distance\n";

        int reachable = 0;
        for (int v = 0; v < g_data.max_n_v; ++v) {
            int gd = tree.dis(v);
            int gp = tree.get_parent(v);
            if (gd != INF_DIST) {
                reachable++;
                gpu_out << v << " " << gp << " " << gd << "\n";
            } else {
                gpu_out << v << " -1 INF\n";
            }

            int cd = dij.dist[v];
            int cp = dij.parent[v];
            if (cd != INF_DIST) {
                dij_out << v << " " << cp << " " << cd << "\n";
            } else {
                dij_out << v << " -1 INF\n";
            }
        }
        gpu_out.close();
        dij_out.close();

        std::cout << "\nSummary across " << host_batches.size() << " batches:\n"
                  << "  Total GPU update_sssp time : " << total_gpu_ms << " ms\n"
                  << "  Total CPU Dijkstra time    : " << total_cpu_ms << " ms\n"
                  << "  Reachable vertices         : " << reachable << "/" << g_data.max_n_v << "\n"
                  << "  Verification status        : "
                  << (total_mismatches == 0 ? "PASS (0 mismatches)" : "FAIL (" + std::to_string(total_mismatches) + " total mismatches)")
                  << "\n"
                  << "  Files written              : result/" << out_prefix << "_timings.csv, "
                  << "result/" << out_prefix << "_output.txt, "
                  << "result/" << out_prefix << "_dijkstra.txt\n";

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    } catch (const std::exception& ex) {
        std::cerr << "Fatal: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
