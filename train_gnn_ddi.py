# -*- coding: utf-8 -*-
import torch
import torch_geometric.transforms as T

from util.hook import set_vertex_map, set_updated_vertex_map
from util.logger import Logger
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models import GAT, GCN, SAGE, LinkPredictor
from util import train_test_ddi, train_decorator
from util.global_variable import *
from util.other import norm_adj, transform_adj_matrix, transform_matrix_2_binary, \
    store_updated_list_and_adj_matrix
from subprocess import call
from tensorboardX import SummaryWriter


def main():
    print(args)
    writer = SummaryWriter()

    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 从ogb用于边预测的数据集中获取ddi数据集，ddi数据集是一个药物相互作用数据集，边的含义是：两种药物一起使用有相互作用
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())

    data = dataset[0]
    # 将邻接矩阵正则化
    adj_matrix = norm_adj(data.adj_t) if args.call_neurosim else None
    print(type(adj_matrix))
    # 获取ddi数据集的邻接矩阵，格式为SparseTensor
    adj_t = data.adj_t
    adj_origin = data.adj_t.to(device)

    # run_recorder.record('', 'adj_dense.csv', data.adj_t.to_dense().numpy(), delimiter=',', fmt='%s')

    # 获取词嵌入数量
    if args.use_cluster:
        cluster_label, embedding_num, adj_matrix, adj_t = transform_adj_matrix(data, device)
    else:
        embedding_num = data.adj_t.size(0)
        cluster_label = None

    if args.call_neurosim:
        adj_coo = adj_matrix.coo()
        adj_stack = torch.stack([adj_coo[0].to(torch.int), adj_coo[1].to(torch.int), adj_coo[2]])
    else:
        adj_stack = None

    # 获取顶点特征更新列表
    updated_vertex, vertex_pointer = store_updated_list_and_adj_matrix(adj_t=adj_t, adj_binary=adj_stack)
    if vertex_pointer is not None:
        set_vertex_map(vertex_pointer)
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
                    recorder=run_recorder).to(device)
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

    if args.percentile != 0:
        train_dec.bind_update_hook(model)

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
