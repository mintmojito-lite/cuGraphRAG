#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../include/graph_store.h"

namespace py = pybind11;

bool gpu_available = false; // No GPU for now

PYBIND11_MODULE(cugraphrag, m) {
    py::class_<GraphStore>(m, "GraphStore")
        .def(py::init<int, int, int>())
        .def("load_graph", [](GraphStore &self,
                              py::array_t<int> row_ptr,
                              py::array_t<int> col_idx,
                              py::array_t<float> embeddings) {

            std::vector<int> row(row_ptr.size());
            std::vector<int> col(col_idx.size());
            std::vector<float> emb(embeddings.size());

            std::memcpy(row.data(), row_ptr.data(), row_ptr.size() * sizeof(int));
            std::memcpy(col.data(), col_idx.data(), col_idx.size() * sizeof(int));
            std::memcpy(emb.data(), embeddings.data(), embeddings.size() * sizeof(float));

            self.loadGraph(row, col, emb);
        })
        .def("query", [&](GraphStore &self,
                          py::array_t<float> query_vec,
                          float threshold) {

            py::print("⚠ CPU fallback active — no GPU detected");

            // Import CPU fallback module
            py::module cpu = py::module::import("cpu_fallback");
            auto cpu_fn = cpu.attr("cpu_dynamic_prune");

            auto result = cpu_fn(
                py::array(self.h_row_ptr.size(), self.h_row_ptr.data()),
                py::array(self.h_col_idx.size(), self.h_col_idx.data()),
                py::array({self.num_nodes, self.dim}, self.h_embeddings.data()),
                query_vec,
                threshold
            );

            return result;
        });

    m.doc() = "cuGraphRAG CPU-only runtime";
}
