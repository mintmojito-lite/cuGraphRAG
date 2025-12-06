#include <math.h>
#include <cuda_runtime.h>
#include "../include/graph_store.h"

#ifndef DIM
#define DIM 128 // Default embedding dimension
#endif

extern "C" __global__
void dynamicPruneKernel(
    const int * __restrict__ row_ptr,
    const int * __restrict__ col_idx,
    const float * __restrict__ embeddings,
    const float * __restrict__ query,
    float threshold,
    bool * __restrict__ keep,
    int num_nodes
) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    if (!keep[node]) return;

    float dot = 0.0f, norm_node = 0.0f, norm_q = 0.0f;

    #pragma unroll
    for (int i = 0; i < DIM; i++) {
        float v = embeddings[node * DIM + i];
        float q = query[i];
        dot += v * q;
        norm_node += v * v;
        norm_q += q * q;
    }

    float cosine = dot / (sqrtf(norm_node) * sqrtf(norm_q));

    // If irrelevant — prune early
    if (cosine < threshold) {
        keep[node] = false;
        return;
    }

    // Add neighbors to keep-list
    int start = row_ptr[node];
    int end = row_ptr[node + 1];
    for (int i = start; i < end; i++) {
        int nb = col_idx[i];
        keep[nb] = true;
    }
}
