import argparse
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch_geometric.transforms as T
from models import GCN, SAGE, QG, QW, C
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from util import train_decorator
from util.definition import DropMode, ClusterAlg
from util.global_variable import run_recorder
from util.hook import set_vertex_map, set_updated_vertex_map
from util.logger import Logger
import numpy as np
from util.global_variable import args
from util.other import get_vertex_cluster, map_adj_to_cluster_adj, norm_adj, dec2bin, get_updated_list
from subprocess import call
from tensorboardX import SummaryWriter
from torch_sparse import SparseTensor

from util.train_decorator import TrainDecorator


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, data, split_edge, optimizer, batch_size, train_decorator: TrainDecorator,
          cur_epoch=0, cluster_label=None):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0
    for i, perm in enumerate(DataLoader(range(source_edge.size(0)), batch_size, shuffle=True)):
        # 量化权重
        train_decorator.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        if args.call_neurosim:
            train_decorator.bind_hooks(model, i, cur_epoch)
        optimizer.zero_grad()

        h = model(data.x, data.adj_t)

        src, dst = source_edge[perm], target_edge[perm]
        # 将原图上的边转换为聚类后图上的边
        if args.use_cluster:
            src = cluster_label[src]
            dst = cluster_label[dst]

        pos_out = predictor(h[src], h[dst])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, data.num_nodes, src.size(),
                                dtype=torch.long, device=h.device)
        # 将原图上的边转换为聚类后图上的边
        if args.use_cluster:
            dst_neg = cluster_label[dst_neg]

        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        # 量化梯度
        train_decorator.quantify_activation(model)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        # 清楚钩子
        if args.call_neurosim:
            train_decorator.clear_hooks(model, i, cur_epoch)

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, cluster_label=None):
    predictor.eval()

    h = model(data.x, data.adj_t)

    def test_split(split):
        source = split_edge[split]['source_node'].to(h.device)
        target = split_edge[split]['target_node'].to(h.device)
        target_neg = split_edge[split]['target_node_neg'].to(h.device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            if args.use_cluster:
                src = cluster_label[src]
                dst = cluster_label[dst]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            if args.use_cluster:
                src = cluster_label[src]
                dst_neg = cluster_label[dst_neg]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation2 (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
    data = data.to(device)

    weight_quantification = lambda x, scale: QW(x, args.bl_weight, scale)
    grad_quantiication = lambda x: QG(x, args.bl_grad, args.bl_rand, args.lr)
    grad_clip = lambda x: C(x, args.bl_weight)

    train_dec = train_decorator.TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)
    # 获取词嵌入数量
    cluster_label = None
    # 将邻接矩阵正则化
    adj_matrix = norm_adj(data.adj_t).to_dense().numpy()
    adj_t = data.adj_t
    if args.use_cluster:
        cluster_label = get_vertex_cluster(data.adj_t.to_dense().numpy(), ClusterAlg(args.cluster_alg))
        adj_dense = map_adj_to_cluster_adj(data.adj_t.to_dense().numpy(), cluster_label)
        adj_t = SparseTensor.from_dense(adj_dense)
        data.adj_t = adj_t
        adj_matrix = norm_adj(adj_t).to_dense().numpy()
        embedding_num = adj_matrix.shape[0]
        cluster_label = torch.from_numpy(cluster_label)
        data.x = torch.nn.Embedding(embedding_num, data.num_features).to(device)

    # 转换为2进制
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

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder, adj_activity=activity).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-citation2')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, split_edge, optimizer,
                         args.batch_size, train_decorator=train_dec, cur_epoch=epoch,
                         cluster_label=cluster_label)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')

            if epoch % args.eval_steps == 0:
                result = test(model, predictor, data, split_edge, evaluator,
                              args.batch_size, cluster_label=cluster_label)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')

        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)
    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()
