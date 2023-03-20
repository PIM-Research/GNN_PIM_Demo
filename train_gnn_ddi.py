import torch
import torch_geometric.transforms as T
from util.logger import Logger
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from models import GAT, GCN, SAGE, LinkPredictor, QW, QG, C
from util import recorder, train_test_ddi, train_decorator
from util.global_variable import *
from util.other import norm_adj, dec2bin
import numpy as np
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor
import os
from subprocess import call


def main():
    print(args)
    if not os.path.exists('./NeuroSim_Results_Each_Epoch'):
        os.makedirs('./NeuroSim_Results_Each_Epoch')
    # 定义量化权重和梯度的lambda函数以及权重clip函数
    weight_quantification = lambda x, scale: QW(x, args.bl_weight, scale)
    grad_quantiication = lambda x: QG(x, args.bl_grad, args.bl_rand, args.lr)
    grad_clip = lambda x: C(x, args.bl_weight)

    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # 从ogb用于边预测的数据集中获取ddi数据集，ddi数据集是一个药物相互作用数据集，边的含义是：两种药物一起使用有相互作用
    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    # adj_matrix_tensor = add_self_loops(data.adj_t)
    # adj_matrix_size = adj_matrix_tensor[0].shape[0]
    # adj_matrix = SparseTensor(row=adj_matrix_tensor[0][0], col=adj_matrix_tensor[0][1],
    #                           sparse_sizes=(adj_matrix_size, adj_matrix_size)).to_dense().numpy()
    #adj_matrix = norm_adj(data.adj_t).to_dense().numpy()
    #run_recorder.record('', 'adj_matrix.csv', adj_matrix, delimiter=',', fmt='%d')
    adj_matrix = norm_adj(data.adj_t).to_dense().numpy()
    adj_binary = np.zeros([adj_matrix.shape[0], adj_matrix.shape[1] * args.bl_activate], dtype=np.str_)
    adj_binary_col, scale = dec2bin(adj_matrix, args.bl_activate)
    for i, b in enumerate(adj_binary_col):
        adj_binary[:, i::args.bl_activate] = b
    run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    activity = np.sum(adj_matrix.astype(np.float), axis=None) / np.size(adj_matrix)
    # 获取ddi数据集的邻接矩阵，格式为SparseTensor

    adj_t = data.adj_t.to(device)
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
    emb = torch.nn.Embedding(data.adj_t.size(0),
                             args.hidden_channels).to(device)

    # 实例化一个预测器，给定两个顶点的特征向量，预测这两个顶点之间是否存在边
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(emb.parameters()) +
            list(predictor.parameters()), lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train_test_ddi.train(model, predictor, emb.weight, adj_t, split_edge,
                                        optimizer, args.batch_size, train_decorator=train_dec, cur_epoch=epoch)

            if epoch % args.eval_steps == 0:
                results = train_test_ddi.test(model, predictor, emb.weight, adj_t, split_edge,
                                              evaluator, args.batch_size)
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
                    print('---') 
            call(["chmod", "o+x", run_recorder.bootstrap_path])
            call(["/bin/bash", run_recorder.bootstrap_path])
		
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

       # model.record_cost()
	
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
