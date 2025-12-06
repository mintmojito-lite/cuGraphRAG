import numpy as np
import cugraphrag as gr

row_ptr = np.array([0, 2, 3, 4, 4, 4], dtype=np.int32)
col_idx = np.array([1, 2, 3, 4], dtype=np.int32)
dim = 128

emb = np.random.rand(5 * dim).astype(np.float32)
query = np.random.rand(dim).astype(np.float32)

g = gr.GraphStore(5, 4, dim)
g.load_graph(row_ptr, col_idx, emb)

result = g.query(query, 0.6)
print("CPU result mask:", result)
