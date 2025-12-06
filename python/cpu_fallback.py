import numpy as np

def cpu_dynamic_prune(row_ptr, col_idx, embeddings, query, threshold):
    num_nodes = embeddings.shape[0]
    keep = np.ones(num_nodes, dtype=bool)

    query_norm = np.linalg.norm(query)

    for node in range(num_nodes):
        if not keep[node]:
            continue

        v = embeddings[node]
        score = np.dot(v, query) / (np.linalg.norm(v) * query_norm)

        if score < threshold:
            keep[node] = False
            continue

        start = row_ptr[node]
        end = row_ptr[node + 1]
        for i in range(start, end):
            keep[col_idx[i]] = True

    return keep
