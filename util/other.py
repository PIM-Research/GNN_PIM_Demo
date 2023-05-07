import os
import random
from typing import Union, Tuple
import scipy.sparse as sp
import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec
from torch_sparse import sum as sparse_sum, fill_diag, mul, SparseTensor
from .hook import set_min_dis_vertex
from models import Q
from .definition import DropMode, ClusterAlg, MappingAlg, ClusterBasis, EmbeddingAlg
from math import ceil, floor
from sklearn import cluster
from .global_variable import args, run_recorder
import torch

vertex_pointer = None


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
    np.add.at(vertex_deg, des_vertex.cpu().numpy(), 1)
    return vertex_deg


def map_data(data: np.ndarray, array_size, vertex_num):
    data_mapped = np.zeros(len(data), dtype=data.dtype)
    region_num = floor(vertex_num / array_size)
    for i in range(0, region_num):
        data_mapped[i * array_size:(i + 1) * array_size] = data[i:region_num * array_size:region_num]
    data_mapped[region_num * array_size:] = data[region_num * array_size:]
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
    vertex_map = map_data(vertex_list, array_size, vertex_num)
    return vertex_deg_global, vertex_map


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
    # 若按照源节点聚类，则转置邻接矩阵
    if ClusterBasis(args.cluster_basis) is ClusterBasis.SRC:
        adj_dense = np.transpose(adj_dense)
    if cluster_alg is ClusterAlg.DBSCAN:
        # 采用DBSCAN算法聚类
        dbscan = cluster.DBSCAN(eps=np.sqrt(adj_dense.shape[0] * args.eps), min_samples=args.min_samples)
        cluster_label = dbscan.fit_predict(adj_dense)
    elif cluster_alg is ClusterAlg.K_MEANS:
        # 采用K_MEANS算法聚类
        k_means = cluster.KMeans(n_clusters=round(adj_dense.shape[0] * args.kmeans_clusters), n_init=args.n_init)
        cluster_label = k_means.fit_predict(adj_dense)
    elif cluster_alg is ClusterAlg.SC:
        # 采用谱聚类算法聚类
        sc = cluster.SpectralClustering(n_clusters=round(adj_dense.shape[0] * args.kmeans_clusters), random_state=0,
                                        n_init=args.n_init)
        cluster_label = sc.fit_predict(adj_dense)
    else:
        cluster_label = np.arange(adj_dense.shape[0])
    cluster_num = np.max(cluster_label) + 1
    # 将被DBSCAN算法评定为噪声的点作为单独的一类
    vertex_noise = cluster_label == -1
    cluster_append = np.random.permutation(np.arange(cluster_num, cluster_num + cluster_label[vertex_noise].shape[0]))
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


def map_adj_to_cluster_adj(adj_sparse: SparseTensor, cluster_label: np.ndarray) -> SparseTensor:
    adj_sparse = adj_sparse.coo()
    # 获取簇的数量
    cluster_num = np.max(cluster_label) + 1
    # 获取行和列所属的簇
    rows_cluster = cluster_label[adj_sparse[0]]
    cols_cluster = cluster_label[adj_sparse[1]]
    # 获取行和列不在同一簇中的元素
    if args.add_self_loop is False:
        mask = rows_cluster != cols_cluster
        rows_cluster, cols_cluster = rows_cluster[mask], cols_cluster[mask]
    # 将稀疏张量转换成稠密矩阵，并将它的值赋给对应的簇之间的位置
    rows_cluster = torch.from_numpy(rows_cluster)
    cols_cluster = torch.from_numpy(cols_cluster)
    values = torch.ones(rows_cluster.shape[0])
    cluster_adj = SparseTensor(row=rows_cluster, col=cols_cluster, value=values,
                               sparse_sizes=(cluster_num, cluster_num))
    # 将稀疏张量转换成COO格式
    cluster_adj = cluster_adj.coalesce()
    return cluster_adj


def transform_adj_matrix(data, device):
    # 进行图嵌入
    adj_dense = map_node_to_vec(data.adj_t)
    cluster_label = get_vertex_cluster(adj_dense, ClusterAlg(args.cluster_alg))
    adj_t = map_adj_to_cluster_adj(data.adj_t, cluster_label)
    adj_t_coo = adj_t.coo()
    adj_t_coo = torch.stack([adj_t_coo[0], adj_t_coo[1], adj_t_coo[2]])
    run_recorder.record('', 'cluster_adj_dense.csv', adj_t_coo, delimiter=',', fmt='%s')
    run_recorder.record('', 'cluster_label.csv', cluster_label, delimiter=',', fmt='%s')
    adj_t.value = None  # 将value属性置为None
    adj_matrix = norm_adj(adj_t)
    # 量化邻接矩阵
    if args.bl_activate != -1:
        adj_matrix = quantify_adj(adj_matrix, args.bl_activate)
    embedding_num = max(adj_matrix.size(dim=0), adj_matrix.size(dim=1))
    cluster_label = torch.from_numpy(cluster_label).long().to(device)
    print(embedding_num)
    return cluster_label, embedding_num, adj_matrix, adj_t


def transform_matrix_2_binary(adj_matrix):
    adj_binary = None
    activity = 0
    if args.call_neurosim:
        adj_binary = np.zeros([adj_matrix.shape[0], adj_matrix.shape[1] * args.bl_activate], dtype=np.str_)
        adj_binary_col, scale = dec2bin(adj_matrix, args.bl_activate)
        for i, b in enumerate(adj_binary_col):
            adj_binary[:, i::args.bl_activate] = b
        activity = np.sum(adj_binary.astype(np.float64), axis=None) / np.size(adj_binary)
    return adj_binary, activity


def store_updated_list_and_adj_matrix(adj_t, adj_binary):
    if args.call_neurosim:
        adj_binary = filter_edges_by_avg(adj_binary)
        adj_binary = adj_binary.cpu().coo()
    else:
        adj_binary = None
    drop_mode = DropMode(args.drop_mode)
    vertex_map = None
    if args.percentile != 0:
        if drop_mode == DropMode.GLOBAL:
            updated_vertex, vertex_map = get_updated_list(adj_t, args.percentile, args.array_size, drop_mode)
            if args.call_neurosim:
                run_recorder.record_acc_vertex_map('', 'adj_matrix.csv', adj_binary, vertex_map, delimiter=',',
                                                   fmt='%s')
        else:
            updated_vertex = get_updated_list(adj_t, args.percentile, args.array_size, drop_mode)
            if args.call_neurosim:
                adj_binary = torch.stack([adj_binary[0], adj_binary[1], adj_binary[2]]) \
                    if adj_binary[2] is not None else torch.stack([adj_binary[0], adj_binary[1]])
                run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    else:
        updated_vertex = np.ones(max(adj_t.size(dim=0), adj_t.size(dim=1)))
        if args.call_neurosim:
            adj_binary = torch.stack([adj_binary[0], adj_binary[1], adj_binary[2]]) \
                if adj_binary[2] is not None else torch.stack([adj_binary[0], adj_binary[1]])
            run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    if args.call_neurosim:
        run_recorder.record('', 'updated_vertex.csv', updated_vertex.transpose(), delimiter=',', fmt='%d')
    return updated_vertex, vertex_map


def record_net_structure(embedding_num, input_channels, hidden_channels, output_channels, num_layers):
    net_dir = './NeuroSIM/NetWork.csv'
    if os.path.exists(net_dir):
        os.remove(net_dir)
    with open(net_dir, 'w') as f:
        f.write(f'1,1,{input_channels},1,1,{hidden_channels},0,1\n')
        f.write(f'1,1,{embedding_num},1,1,{hidden_channels},0,1\n')
        for i in range(num_layers - 2):
            f.write(f'1,1,{hidden_channels},1,1,{hidden_channels},0,1\n')
            f.write(f'1,1,{embedding_num},1,1,{hidden_channels},0,1\n')
        f.write(f'1,1,{hidden_channels},1,1,{output_channels},0,1\n')
        f.write(f'1,1,{embedding_num},1,1,{output_channels},0,1\n')


def quantify_adj(adj: SparseTensor, n):
    row, col, value = adj.coo()
    value = Q(value, n)
    return SparseTensor(row=row, col=col, value=value)


def get_dense(adj: SparseTensor):
    row, col, value = adj.coo()
    node_num = max(adj.size(dim=0), adj.size(dim=1))
    adj_dense = np.zeros((node_num, node_num))
    adj_dense[row, col] = value if value is not None else 1
    return adj_dense


def filter_edges(adj: SparseTensor, vertex_filter):
    # 找到目的顶点存在于A中的边的索引
    vertex_num = max(adj.size(dim=0), adj.size(dim=1))
    adj = adj.coo()
    mask = torch.isin(adj[1], vertex_filter)
    # 过滤出符合条件的边的索引
    row = adj[0][mask]
    col = adj[1][mask]
    value = adj[2][mask] if adj[2] is not None else None
    size = (vertex_num, vertex_num)
    # 返回结果
    return SparseTensor(row=row, col=col, value=value, sparse_sizes=size)


def store_adj_matrix(adj: SparseTensor):
    if args.call_neurosim:
        adj_coo = adj.cpu().coo()
        drop_mode = DropMode(args.drop_mode)
        if drop_mode == DropMode.GLOBAL:
            assert vertex_pointer is not None
            run_recorder.record_acc_vertex_map('', 'adj_matrix.csv', adj_coo, vertex_pointer, delimiter=',',
                                               fmt='%s')
        else:
            run_recorder.record('', 'adj_matrix.csv', adj_coo, delimiter=',', fmt='%s')


def store_updated_list(adj_t):
    drop_mode = DropMode(args.drop_mode)
    vertex_map = None
    if args.percentile != 0:
        if drop_mode == DropMode.GLOBAL:
            updated_vertex, vertex_map = get_updated_list(adj_t, args.percentile, args.array_size, drop_mode)
            set_vertex_pointer(vertex_map)
        else:
            updated_vertex = get_updated_list(adj_t, args.percentile, args.array_size, drop_mode)
    else:
        updated_vertex = np.ones(max(adj_t.size(dim=0), adj_t.size(dim=1)))
    if args.call_neurosim:
        run_recorder.record('', 'updated_vertex.csv', updated_vertex.transpose(), delimiter=',', fmt='%d')
    return updated_vertex, vertex_map


def set_vertex_pointer(vertex_map):
    global vertex_pointer
    vertex_pointer = vertex_map


def filter_edges_by_avg(adj: SparseTensor):
    adj_coo = adj.coo()
    vertex_deg = get_vertex_deg(adj)
    deg_avg = np.sum(vertex_deg) / vertex_deg.shape[0]
    min_dis_index = np.argmin(np.abs(vertex_deg - deg_avg))
    set_min_dis_vertex(min_dis_index)
    mask = adj_coo[1] == min_dis_index
    row, col = adj_coo[0][mask], adj_coo[1][mask]
    value = adj_coo[2][mask] if adj_coo[2] is not None else None
    return SparseTensor(row=row, col=col, value=value)


def map_node_to_vec(adj: SparseTensor):
    # 将SparseTensor类型转换成COO稀疏矩阵的形式
    coo = adj.coo()
    value = torch.ones(size=coo[0].shape)
    # 将COO稀疏矩阵转换成scipy.sparse中的稀疏矩阵类型
    sparse_matrix = sp.coo_matrix((value.tolist(), (coo[0].tolist(), coo[1].tolist())))
    # 将scipy.sparse中的稀疏矩阵类型转换成networkx中的图类型
    if args.di_graph is True:
        graph = nx.from_scipy_sparse_matrix(sparse_matrix, create_using=nx.DiGraph())
    else:
        graph = nx.from_scipy_sparse_array(sparse_matrix)
    if args.embedding_alg is EmbeddingAlg.NODE2VEC:
        # 初始化node2vec模型
        node2vec = Node2Vec(graph, dimensions=32, walk_length=3, num_walks=40, workers=4)
        # 训练模型
        vec = node2vec.fit(window=3, min_count=1, batch_words=4).wv.vectors
    else:
        # 执行随机游走
        num_walks = args.num_walks  # 游走次数
        walk_length = args.walk_length  # 游走长度
        walks = []
        for _ in range(num_walks):
            for node in graph.nodes():
                walk = random_walk(graph, node, walk_length)
                walks.append(walk)
        model = Word2Vec(vector_size=args.embedding_size, window=args.walk_window, sg=1, hs=0, negative=10, alpha=0.03,
                         min_alpha=0,
                         seed=14)  # 训练Word2Vec模型
        model.build_vocab(walks, progress_per=2)
        model.train(walks, total_examples=model.corpus_count, epochs=50, report_delay=1)
        # 获取节点的嵌入向量
        vec = model.wv.vectors
    return vec


def random_walk(graph, start_node, length):
    walk = [start_node]
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(walk[-1]))
        if neighbors:
            walk.append(random.choice(neighbors))
        else:
            break
    return walk
