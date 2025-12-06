#include "../include/graph_store.h"
#include <stdexcept>
#include <iostream>

GraphStore::GraphStore(int num_nodes, int num_edges, int dim)
    : num_nodes(num_nodes), num_edges(num_edges), dim(dim),
      d_row_ptr(nullptr), d_col_idx(nullptr), d_embeddings(nullptr) {}

GraphStore::~GraphStore() {
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_idx) cudaFree(d_col_idx);
    if (d_embeddings) cudaFree(d_embeddings);
}

void GraphStore::loadGraph(const std::vector<int>& row_ptr,
                           const std::vector<int>& col_idx,
                           const std::vector<float>& embeddings)
{
    if (row_ptr.size() != num_nodes + 1)
        throw std::runtime_error("row_ptr size mismatch!");

    if (col_idx.size() != num_edges)
        throw std::runtime_error("col_idx size mismatch!");

    if (embeddings.size() != num_nodes * dim)
        throw std::runtime_error("embeddings size mismatch!");

    // Allocate GPU memory
    cudaMalloc(&d_row_ptr, (num_nodes + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, num_edges * sizeof(int));
    cudaMalloc(&d_embeddings, num_nodes * dim * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_row_ptr, row_ptr.data(),
               (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_col_idx, col_idx.data(),
               num_edges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_embeddings, embeddings.data(),
               num_nodes * dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    std::cout << "Graph loaded to GPU successfully!" << std::endl;
}
extern "C" void runDynamicPrune(
    GraphStore* graph,
    const float* d_query,
    float threshold,
    bool* d_keep
) {
    int threads = 256;
    int blocks = (graph->num_nodes + threads - 1) / threads;

    dynamicPruneKernel<<<blocks, threads>>>(
        graph->d_row_ptr,
        graph->d_col_idx,
        graph->d_embeddings,
        d_query,
        threshold,
        d_keep,
        graph->num_nodes
    );
    cudaDeviceSynchronize();
}
