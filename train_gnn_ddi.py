# -*- coding: utf-8 -*-
import torch
import torch_geometric.transforms as T
from util.logger import Logger
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models import GAT, GCN, SAGE, LinkPredictor, QW, QG, C
from util import train_test_ddi, train_decorator
from util.global_variable import *
from util.other import norm_adj, dec2bin, get_updated_list, get_vertex_cluster, map_adj_to_cluster_adj
from util.definition import DropMode, ClusterAlg
from util.hook import set_vertex_map, set_updated_vertex_map, hook_forward_set_grad_zero
import numpy as np
from subprocess import call
from tensorboardX import SummaryWriter
from torch_sparse import SparseTensor


def main():
    print(args)
    writer = SummaryWriter()

    # 定义量化权重和梯度的lambda函数以及权重clip函数
    weight_quantification = lambda x, scale: QW(x, args.bl_weight, scale)
    grad_quantiication = lambda x: QG(x, args.bl_grad, args.bl_rand, args.lr)
    grad_clip = lambda x: C(x, args.bl_weight)

    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 从ogb用于边预测的数据集中获取ddi数据集，ddi数据集是一个药物相互作用数据集，边的含义是：两种药物一起使用有相互作用
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())

    data = dataset[0]
    # 将邻接矩阵正则化
    adj_matrix = norm_adj(data.adj_t).to_dense().numpy()
    # 获取ddi数据集的邻接矩阵，格式为SparseTensor
    adj_t = data.adj_t
    adj_origin = data.adj_t.to(device)

    run_recorder.record('', 'adj_dense.csv', data.adj_t.to_dense().numpy(), delimiter=',', fmt='%s')

    # 获取词嵌入数量
    embedding_num = data.adj_t.size(0)
    cluster_label = None
    if args.use_cluster:
        cluster_label = get_vertex_cluster(data.adj_t.to_dense().numpy(), ClusterAlg(args.cluster_alg))
        adj_dense = map_adj_to_cluster_adj(data.adj_t.to_dense().numpy(), cluster_label)
        run_recorder.record('', 'cluster_adj_dense.csv', adj_dense, delimiter=',', fmt='%s')
        run_recorder.record('', 'cluster_label.csv', cluster_label, delimiter=',', fmt='%s')
        adj_t = SparseTensor.from_dense(adj_dense)
        adj_t.value = None  # 将value属性置为None
        adj_t = adj_t.coalesce()
        adj_matrix = norm_adj(adj_t).to_dense().numpy()
        embedding_num = adj_matrix.shape[0]
        cluster_label = torch.from_numpy(cluster_label).long().to(device)
        print(embedding_num)

    # 转换为2进制
    adj_binary = None
    activity = 0
    if args.call_neurosim:
        adj_binary = np.zeros([adj_matrix.shape[0], adj_matrix.shape[1] * args.bl_activate], dtype=np.str_)
        adj_binary_col, scale = dec2bin(adj_matrix, args.bl_activate)
        for i, b in enumerate(adj_binary_col):
            adj_binary[:, i::args.bl_activate] = b
        activity = np.sum(adj_binary.astype(np.float64), axis=None) / np.size(adj_binary)

    # 获取顶点特征更新列表
    drop_mode = DropMode(args.drop_mode)
    if args.percentile != 0:
        if drop_mode == DropMode.GLOBAL:
            updated_vertex, vertex_pointer = get_updated_list(adj_t, args.percentile, args.array_size,
                                                              drop_mode)
            set_vertex_map(vertex_pointer)
            if args.call_neurosim:
                run_recorder.record_acc_vertex_map('', 'adj_matrix.csv', adj_binary, vertex_pointer, delimiter=',',
                                                   fmt='%s')
        else:
            updated_vertex = get_updated_list(adj_t, args.percentile, args.array_size, drop_mode)
            if args.call_neurosim:
                run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    else:
        updated_vertex = np.ones(max(adj_t.size(dim=0), adj_t.size(dim=1)))
        if args.call_neurosim:
            run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    if args.call_neurosim:
        run_recorder.record('', 'updated_vertex.csv', updated_vertex.transpose(), delimiter=',', fmt='%d')
    set_updated_vertex_map(updated_vertex)

    # 将边数据集拆分为训练集，验证集，测试集，其中验证集和测试集有两个属性，edge代表图中存在的边（正边），edge_neg代表图中不存在的边（负边）
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    # 实例化model，从GCN，GraphSage，GAT中选择一个
    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    elif args.use_gcn:
        model = GCN(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder, adj_activity=activity).to(device)
    else:
        model = GAT(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    # 通过词嵌入的方式获得顶点的特征向量（emb.weight），即IFM，下面的参数中，第一个代表向量数量，第二个代表向量维度，emb.weight就是可学习的权重
    emb = torch.nn.Embedding(embedding_num, args.hidden_channels).to(device)

    # 实例化一个预测器，给定两个顶点的特征向量，预测这两个顶点之间是否存在边
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    # 将邻接矩阵放到设备上
    adj_t = adj_t.to(device)

    # 添加钩子使得drop掉的顶点特征不更新
    if args.percentile != 0:
        for index, (name, layer) in enumerate(model.convs.named_children()):
            for index_c, (name_c, layer_c) in enumerate(layer.gcn_conv.named_children()):
                layer_c.register_forward_hook(hook_forward_set_grad_zero)

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train_test_ddi.train(model, predictor, emb.weight, adj_t, split_edge, optimizer, args.batch_size,
                                        train_decorator=train_dec, cur_epoch=epoch,
                                        cluster_label=cluster_label, adj_origin=adj_origin)
            writer.add_scalar('Loss', loss, epoch)

            if epoch % args.eval_steps == 0:
                results = train_test_ddi.test(model, predictor, emb.weight, adj_t, split_edge, evaluator,
                                              args.batch_size, cluster_label=cluster_label)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                        writer.add_scalar(f'{key} Train accuracy', 100 * train_hits, epoch)
                        writer.add_scalar(f'{key} Valid accuracy', 100 * valid_hits, epoch)
                        writer.add_scalar(f'{key} Test accuracy', 100 * test_hits, epoch)
                    print('---')
            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, key=key)

    # model.record_cost()

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)


if __name__ == "__main__":
    main()
