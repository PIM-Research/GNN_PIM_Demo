import numpy as np
from torch_sparse import sum as sparse_sum, fill_diag, mul, SparseTensor
from .definition import DropMode, ClusterAlg, MappingAlg
from math import ceil, floor
from sklearn import cluster
from .global_variable import args
import torch


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
    np.add.at(vertex_deg, des_vertex, 1)
    return vertex_deg


def map_data(data: np.ndarray, array_size, vertex_num):
    data_mapped = np.zeros(len(data), dtype=data.dtype)
    region_num = floor(vertex_num / array_size)
    for i in range(0, region_num):
        data_mapped[i * array_size:(i + 1) * array_size] = data[i:region_num * array_size:region_num]
    data_mapped[region_num * array_size:] = data[region_num * array_size:]
    # index = 0
    # for i in range(0, array_num):
    #     if (array_size - 1) * array_num + i < vertex_num:
    #         data_mapped[i * array_size:(i + 1) * array_size] = data[i::array_num]
    #     elif (i + 1) * array_size < vertex_num:
    #         data_mapped[i * array_size:(i + 1) * array_size - 1] = data[i::array_num]
    #         data_mapped[(i + 1) * array_size - 1] = data[
    #             (vertex_num % array_size + index + 1) * array_num - 1]
    #         index += 1
    #     else:
    #         data_mapped[i * array_size:vertex_num] = data[
    #                                                  i:(vertex_num % array_size * array_num):array_num]
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
    # 将vertex_deg_dec分成array_size个区间，每个区间最多有array_num个顶点，每个crossbar按行顺序依次从每一个区间取出一个顶点进行映射
    vertex_deg_global = map_data(vertex_deg_dec, array_size, vertex_num)
    vertex_pointer = map_data(vertex_list, array_size, vertex_num)
    return vertex_deg_global, vertex_pointer


def get_updated_list(adj: SparseTensor, percentage, array_size, drop_mode: DropMode):
    # 获取原始的顶点度列表、n百分位数，顶点数量，列上需要的crossbar array数量
    vertex_deg = get_vertex_deg(adj)
    vertex_num = vertex_deg.shape[0]
    updated_list = np.zeros(vertex_deg.shape, dtype=np.int_)
    pointer_list = None
    updated_list, pointer_list = get_updated_list_reuse(vertex_deg, vertex_num, percentage, array_size,
                                                        drop_mode)
    if drop_mode is DropMode.GLOBAL:
        return updated_list, pointer_list
    else:
        return updated_list


def get_updated_list_reuse(deg_list, list_size, percentage, array_size, drop_mode: DropMode):
    updated_list = np.zeros(deg_list.shape, dtype=np.int_)
    array_num = ceil(list_size / array_size)
    n_percentile = np.percentile(deg_list, percentage)
    # 根据drop的模式选择对应的方法生成待更新顶点列表
    if drop_mode is DropMode.GLOBAL:  # 每一个crossbar上均匀分布各个度大小的顶点，然后按照全体顶点度的n百分位数选择待更新顶点
        deg_list_global, pointer_list = get_vertex_deg_global(deg_list, array_size)
        updated_list[:] = list(map(lambda x: 1 if x > n_percentile else 0, deg_list_global))
        return updated_list, pointer_list
    elif drop_mode is DropMode.LOCAL:  # 每一个crossbar按原始顶点顺序映射顶点特征，然后当前crossbar上映射顶点的n百分位数选择待更新顶点
        for i in range(0, array_num):
            if (i + 1) * array_size <= list_size:
                n_percentile = np.percentile(deg_list[i * array_size:(i + 1) * array_size], percentage)
                updated_list[i * array_size:(i + 1) * array_size] = list(
                    map(lambda x: 1 if x > n_percentile else 0, deg_list[i * array_size:(i + 1) * array_size]))
            else:
                n_percentile = np.percentile(deg_list[i * array_size:list_size], percentage)
                updated_list[i * array_size:list_size] = list(
                    map(lambda x: 1 if x > n_percentile else 0, deg_list[i * array_size:list_size]))
    elif drop_mode is DropMode.ORIGINAL:  # 每一个crossbar按原始顶点顺序映射顶点特征，然后按照全体顶点度的n百分位数选择待更新顶点
        updated_list[:] = list(map(lambda x: 1 if x > n_percentile else 0, deg_list))
    return updated_list, None


def get_vertex_cluster(adj_dense: np.ndarray, cluster_alg: ClusterAlg):
    if cluster_alg is ClusterAlg.DBSCAN:
        # 创建DBSCAN，并聚类
        dbscan = cluster.DBSCAN(eps=np.sqrt(adj_dense.shape[0] * args.eps), min_samples=args.min_samples)
        cluster_label = dbscan.fit_predict(adj_dense)
    elif cluster_alg is ClusterAlg.K_MEANS:
        # 创建K_MEANS，并聚类
        k_means = cluster.KMeans(n_clusters=round(adj_dense.shape[0] * args.kmeans_clusters), n_init=args.n_init)
        cluster_label = k_means.fit_predict(adj_dense)
    else:
        cluster_label = np.arange(adj_dense.shape[0])
    cluster_num = np.max(cluster_label) + 1
    # 将被DBSCAN算法评定为噪声的点作为单独的一类
    vertex_noise = cluster_label == -1
    cluster_append = np.arange(cluster_num, cluster_num + cluster_label[vertex_noise].shape[0])
    cluster_label[vertex_noise] = cluster_append
    # 返回各个顶点所属聚类标签的列表，顺序是原始顺序
    return cluster_label


def get_cluster_avg_deg(cluster_label: np.ndarray, vertex_deg: np.ndarray):
    cluster_num = len(np.unique(cluster_label))
    cluster_deg_sum = np.zeros(cluster_num)
    cluster_vertex_count = np.zeros(cluster_num)

    # 使用高级索引，快速累加每个聚类的度数和顶点数
    np.add.at(cluster_deg_sum, cluster_label, vertex_deg)
    np.add.at(cluster_vertex_count, cluster_label, 1)

    # 计算每个聚类的平均度数
    cluster_avg_deg = np.divide(cluster_deg_sum, cluster_vertex_count, where=cluster_vertex_count != 0)

    return cluster_avg_deg, cluster_num


def map_adj_to_cluster_adj(adj_dense: np.ndarray, cluster_label: np.ndarray) -> torch.Tensor:
    mapping_alg = MappingAlg(args.mapping_alg)
    # 获取簇的数量
    cluster_num = np.max(cluster_label) + 1
    # 创建稠密矩阵
    cluster_adj = torch.zeros((cluster_num, cluster_num))
    # 根据不同的映射方式将顶点的邻接矩阵转换为类的邻接矩阵
    if mapping_alg is MappingAlg.UNION:
        # 获取行和列所属的簇
        rows_cluster = cluster_label[np.nonzero(adj_dense)[0]]
        cols_cluster = cluster_label[np.nonzero(adj_dense)[1]]
        # 获取行和列不在同一簇中的元素
        mask = rows_cluster != cols_cluster
        rows_cluster, cols_cluster = rows_cluster[mask], cols_cluster[mask]
        # 将稀疏张量转换成稠密矩阵，并将它的值赋给对应的簇之间的位置
        cluster_adj[rows_cluster, cols_cluster] = 1
    elif mapping_alg is MappingAlg.MEAN:
        vertex_deg = np.sum(adj_dense, axis=0)
        cluster_avg_deg = get_cluster_avg_deg(cluster_label, vertex_deg)[0]
        cluster_vertex_num = np.bincount(cluster_label, minlength=cluster_num)
        for label in range(cluster_num):
            if cluster_vertex_num[label] > 1:
                cluster_vertex_index = np.where(cluster_label == label)[0]
                cluster_vertex_deg = vertex_deg[cluster_vertex_index]
                min_index = np.argmin(np.abs(cluster_vertex_deg - cluster_avg_deg[label]))
                represent_vertex = cluster_vertex_index[min_index]
            else:
                represent_vertex = np.where(cluster_label == label)[0]
            rows_cluster = label
            cols_cluster = cluster_label[np.nonzero(adj_dense[represent_vertex])]
            # 获取行和列不在同一簇中的元素
            mask = cols_cluster != rows_cluster
            cols_cluster = cols_cluster[mask]
            # 将稀疏张量转换成稠密矩阵，并将它的值赋给对应的簇之间的位置
            cluster_adj[rows_cluster, cols_cluster] = 1
    return cluster_adj
