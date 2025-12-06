#pragma once
#include <vector>

class GraphStore {
public:
    int num_nodes;
    int num_edges;
    int dim;

    std::vector<int> h_row_ptr;
    std::vector<int> h_col_idx;
    std::vector<float> h_embeddings;

    GraphStore(int num_nodes, int num_edges, int dim)
        : num_nodes(num_nodes), num_edges(num_edges), dim(dim) {}

    void loadGraph(const std::vector<int>& row_ptr,
                   const std::vector<int>& col_idx,
                   const std::vector<float>& embeddings) {
        h_row_ptr = row_ptr;
        h_col_idx = col_idx;
        h_embeddings = embeddings;
    }
};
