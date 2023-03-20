import numpy as np
from torch_sparse import *
from torch_sparse import sum as sparse_sum


def norm_adj(adj_t, add_self_loops=True):
    """
    normalization adj
    """
    if not adj_t.has_value():
        adj_t = adj_t.fill_value(1.)
    if add_self_loops:
        adj_t = fill_diag(adj_t, 1.)
    deg = sparse_sum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
    return adj_t


def dec2bin(x, n):
    y = x.copy()
    out = []
    scale_list = []
    delta = 1.0 / (2 ** (n - 1))
    x_int = x / delta
    base = 2 ** (n - 1)
    y[x_int >= 0] = 0
    y[x_int < 0] = 1
    rest = x_int + base * y
    out.append(y.copy())
    scale_list.append(-base * delta)
    for i in range(n - 1):
        base = base / 2
        y[rest >= base] = 1
        y[rest < base] = 0
        rest = rest - base * y
        out.append(y.copy())
        scale_list.append(base * delta)
    return out, scale_list


def get_n_percentile(adj: SparseTensor, percentage):
    adj_coo = adj.coo()
    des_vertex = adj_coo[1]
    vertex_deg = np.zeros(adj.size(dim=0))
    for i in des_vertex:
        vertex_deg[i] += 1
    n_percentile = np.percentile(vertex_deg, percentage)
    return n_percentile, vertex_deg


def reset_adj_matrix(adj_original: SparseTensor, adj_norm: np.ndarray, percentage):
    n_percentile, vertex_deg = get_n_percentile(adj_original, percentage)
    for i, deg in enumerate(vertex_deg):
        if deg < n_percentile:
            print(i)
            adj_norm[:, i] = 0
    return adj_norm


def get_updated_vertex(adj: SparseTensor, percentage):
    n_percentile, vertex_deg = get_n_percentile(adj, percentage)
    updated_vertex = np.zeros(shape=[vertex_deg.shape[0]])
    for i, deg in enumerate(vertex_deg):
        if deg >= n_percentile:
            updated_vertex[i] = 1
    return updated_vertex
