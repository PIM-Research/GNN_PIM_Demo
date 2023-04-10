import numpy as np
from torch_sparse import sum as sparse_sum, fill_diag, mul, SparseTensor
from .definition import DropMode
from math import ceil


def norm_adj(adj_t, add_self_loops=True):
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


def get_vertex_deg(adj: SparseTensor):
    # 获取COO格式的邻接矩阵
    adj_coo = adj.coo()
    # 获取目的节点列表
    des_vertex = adj_coo[1]
    # 获取源节点和目的节点最大下标
    source_max = adj.size(dim=0)
    des_max = adj.size(dim=1)
    # 获取原始的顶点度列表
    vertex_deg = np.zeros(max(source_max, des_max), dtype=np.int_)
    for i in des_vertex:
        vertex_deg[i] += 1
    return vertex_deg


def map_data(data: np.ndarray, array_size, array_num, vertex_num):
    data_mapped = np.zeros(len(data), dtype=data.dtype)
    index = 0
    for i in range(0, array_num):
        if (array_size - 1) * array_num + i < vertex_num:
            data_mapped[i * array_size:(i + 1) * array_size] = data[i::array_num]
        elif (i + 1) * array_size < vertex_num:
            data_mapped[i * array_size:(i + 1) * array_size - 1] = data[i::array_num]
            data_mapped[(i + 1) * array_size - 1] = data[
                (vertex_num % array_size + index + 1) * array_num - 1]
            index += 1
        else:
            data_mapped[i * array_size:vertex_num] = data[
                                                     i:(vertex_num % array_size * array_num):array_num]
    return data_mapped


def get_vertex_deg_global(vertex_deg, array_size):
    # 获取顶点度列表的降序排序
    index_deg = [(k, v) for k, v in enumerate(vertex_deg)]
    sorted_index_deg = sorted(index_deg, key=lambda x: x[1], reverse=True)
    dict_sorted_index_deg = dict(sorted_index_deg)
    vertex_list = np.array(list(dict_sorted_index_deg.keys()))
    vertex_deg_dec = np.array(list(dict_sorted_index_deg.values()))
    # 获取顶点数
    vertex_num = vertex_deg.shape[0]
    # 获取列上crossbar array数量
    array_num = ceil(vertex_num / array_size)
    # 将vertex_deg_dec分成array_size个区间，每个区间最多有array_num个顶点，每个crossbar按行顺序依次从每一个区间取出一个顶点进行映射
    vertex_deg_global = map_data(vertex_deg_dec, array_size, array_num, vertex_num)
    vertex_pointer = map_data(vertex_list, array_size, array_num, vertex_num)
    return vertex_deg_global, vertex_pointer


def get_updated_vertex_list(adj: SparseTensor, percentage, array_size, drop_mode: DropMode):
    # 获取原始的顶点度列表、n百分位数，顶点数量，列上需要的crossbar array数量
    vertex_deg = get_vertex_deg(adj)
    vertex_num = vertex_deg.shape[0]
    array_num = ceil(vertex_num / array_size)
    updated_vertex = np.zeros(vertex_deg.shape, dtype=np.int_)
    n_percentile = np.percentile(vertex_deg, percentage)
    # 根据drop的模式选择对应的方法生成待更新顶点列表
    if drop_mode is DropMode.GLOBAL:  # 每一个crossbar上均匀分布各个度大小的顶点，然后按照全体顶点度的n百分位数选择待更新顶点
        vertex_deg_global, vertex_pointer = get_vertex_deg_global(vertex_deg, array_size)
        updated_vertex[:] = list(map(lambda x: 1 if x > n_percentile else 0, vertex_deg_global))
        return updated_vertex, vertex_pointer
    elif drop_mode is DropMode.LOCAL:  # 每一个crossbar按原始顶点顺序映射顶点特征，然后当前crossbar上映射顶点的n百分位数选择待更新顶点
        for i in range(0, array_num):
            if (i + 1) * array_size <= vertex_num:
                n_percentile = np.percentile(vertex_deg[i * array_size:(i + 1) * array_size], percentage)
                # print('n_percentile1:', n_percentile)
                updated_vertex[i * array_size:(i + 1) * array_size] = list(
                    map(lambda x: 1 if x > n_percentile else 0, vertex_deg[i * array_size:(i + 1) * array_size]))
            else:
                n_percentile = np.percentile(vertex_deg[i * array_size:vertex_num], percentage)
                # print('n_percentile2:', n_percentile)
                updated_vertex[i * array_size:vertex_num] = list(
                    map(lambda x: 1 if x > n_percentile else 0, vertex_deg[i * array_size:vertex_num]))
    elif drop_mode is DropMode.ORIGINAL:  # 每一个crossbar按原始顶点顺序映射顶点特征，然后按照全体顶点度的n百分位数选择待更新顶点
        updated_vertex[:] = list(map(lambda x: 1 if x > n_percentile else 0, vertex_deg))
    return updated_vertex
