import argparse
from subprocess import call

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torch_geometric.transforms as T

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor

from models import SAGE, GCN, LinkPredictor
from util import train_decorator
from util.definition import ClusterAlg, DropMode
from util.global_variable import args, run_recorder, weight_quantification, grad_clip, grad_quantiication
from util.hook import set_vertex_map, set_updated_vertex_map, hook_forward_set_grad_zero
from util.logger import Logger
from util.other import get_vertex_cluster, map_adj_to_cluster_adj, get_updated_list, dec2bin
import numpy as np

from util.train_decorator import TrainDecorator


def train(model, predictor, data, split_edge, optimizer, batch_size, train_dec: TrainDecorator,
          cur_epoch=0, cluster_label=None):
    if args.call_neurosim:
        train_dec.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    for i, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                                        shuffle=True)):
        # 量化权重
        train_dec.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        if args.call_neurosim:
            train_dec.bind_hooks(model, i, cur_epoch)
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        edge = pos_train_edge[perm].t()
        # 将原图上的边转换为聚类后图上的边
        if args.use_cluster:
            edge[0] = cluster_label[edge[0]]
            edge[1] = cluster_label[edge[1]]
        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.num_nodes, edge.size(), dtype=torch.long,
                             device=h.device)

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        # 量化梯度
        train_dec.quantify_grad(model)
        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        # 清楚钩子
        if args.call_neurosim:
            train_dec.clear_hooks(model, i, cur_epoch)

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, cluster_label=None):
    model.eval()

    h = model(data.x, data.adj_t)

    pos_train_edge = split_edge['train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        if args.use_cluster:
            edge[0] = cluster_label[edge[0]]
            edge[1] = cluster_label[edge[1]]
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        if args.use_cluster:
            edge[0] = cluster_label[edge[0]]
            edge[1] = cluster_label[edge[1]]
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        if args.use_cluster:
            edge[0] = cluster_label[edge[0]]
            edge[1] = cluster_label[edge[1]]
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        if args.use_cluster:
            edge[0] = cluster_label[edge[0]]
            edge[1] = cluster_label[edge[1]]
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        if args.use_cluster:
            edge[0] = cluster_label[edge[0]]
            edge[1] = cluster_label[edge[1]]
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ppa',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.x = data.x.to(torch.float)
    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)
    data = data.to(device)

    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)
    adj_origin = data.adj_t
    # 获取词嵌入数量
    cluster_label = None
    if args.use_cluster:
        cluster_label = get_vertex_cluster(data.adj_t.to_dense().numpy(), ClusterAlg(args.cluster_alg))
        adj_dense = map_adj_to_cluster_adj(data.adj_t.to_dense().numpy(), cluster_label)
        adj_origin = SparseTensor.from_dense(adj_dense)
        data.adj_t = adj_origin.to(device)
        embedding_num = np.max(cluster_label) + 1
        cluster_label = torch.from_numpy(cluster_label)
        data.x = torch.nn.Embedding(embedding_num, data.num_features).to(device).weight
        data.num_nodes = embedding_num
        data.num_edges = torch.sum(adj_dense)
    split_edge = dataset.get_edge_split()

    if args.use_sage:
        adj_binary = np.zeros([data.adj_t.shape[0], data.adj_t.shape[1] * args.bl_activate], dtype=np.str_)
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t
        adj_matrix = data.adj_t.to_dense().numpy()

        # 转换为2进制
        adj_binary = np.zeros([adj_matrix.shape[0], adj_matrix.shape[1] * args.bl_activate], dtype=np.str_)
        adj_binary_col, scale = dec2bin(adj_matrix, args.bl_activate)
        for i, b in enumerate(adj_binary_col):
            adj_binary[:, i::args.bl_activate] = b
        activity = np.sum(adj_binary.astype(np.float64), axis=None) / np.size(adj_binary)

        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate,
                    bl_error=args.bl_error,
                    recorder=run_recorder, adj_activity=activity).to(device).to(device)

    # 获取顶点特征更新列表
    drop_mode = DropMode(args.drop_mode)
    if args.percentile != 0:
        if drop_mode == DropMode.GLOBAL:
            updated_vertex, vertex_pointer = get_updated_list(adj_origin, args.percentile, args.array_size,
                                                              drop_mode)
            set_vertex_map(vertex_pointer)
            if args.call_neurosim:
                run_recorder.record_acc_vertex_map('', 'adj_matrix.csv', adj_binary, vertex_pointer, delimiter=',',
                                                   fmt='%s')
        else:
            updated_vertex = get_updated_list(adj_origin, args.percentile, args.array_size, drop_mode)
            if args.call_neurosim:
                run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    else:
        updated_vertex = np.ones(max(adj_origin.size(dim=0), adj_origin.size(dim=1)))
        if args.call_neurosim:
            run_recorder.record('', 'adj_matrix.csv', adj_binary, delimiter=',', fmt='%s')
    if args.call_neurosim:
        run_recorder.record('', 'updated_vertex.csv', updated_vertex.transpose(), delimiter=',', fmt='%d')
    set_updated_vertex_map(updated_vertex)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    # 添加钩子使得drop掉的顶点特征不更新
    if args.percentile != 0:
        for index, (name, layer) in enumerate(model.convs.named_children()):
            for index_c, (name_c, layer_c) in enumerate(layer.gcn_conv.named_children()):
                layer_c.register_forward_hook(hook_forward_set_grad_zero)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, train_dec=train_dec, cur_epoch=epoch,
                         cluster_label=cluster_label)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, split_edge, evaluator,
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

            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, key=key)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)


if __name__ == "__main__":
    main()
