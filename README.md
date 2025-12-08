cuGraphRAG 

My custom GraphRAG runtime built from scratch in C++.

This isn’t a Python script. This is a native .pyd engine for fast knowledge-graph retrieval. It prunes irrelevant nodes based on embedding scores, saving LLM context budget and boosting speed. CPU fallback works on any machine. GPU acceleration is ready when plugged into CUDA (coming next).

Why I built this 

GraphRAG is powerful but current implementations are slow (Python + NetworkX).
So I built a runtime that actually performs like a system not a demo.

Key Features

Native C++ GraphStore

Dynamic Relevance Pruning

Importable in Python: import cugraphrag

CPU fallback (works everywhere)

CUDA kernels included (GPU soon)

Run Benchmark 
cd examples
python benchmark_rag.py


Example:

⚠ CPU fallback active no GPU detected
Time: 0.0023 sec

Build (Windows)
mkdir build && cd build
cmake .. && cmake --build . --config Release
copy build\Release\cugraphrag*.pyd <your_site_packages_path>\

Roadmap 

GPU dynamic pruning

Token-budget scoring

Explainable path output

Full LLM integration

Built by Bala
learning by building the stuff I wish existed.
