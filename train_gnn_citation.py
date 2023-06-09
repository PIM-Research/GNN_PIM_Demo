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
from util.global_variable import run_recorder, weight_quantification, grad_clip, grad_quantiication
from util.hook import set_vertex_map, set_updated_vertex_map, hook_forward_set_grad_zero
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


def train(model, predictor, data, split_edge, optimizer, batch_size, train_dec: TrainDecorator,
          cur_epoch=0, cluster_label=None):
    if args.call_neurosim:
        train_dec.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    predictor.train()

    source_edge = split_edge['train']['source_node'].to(data.x.device)
    target_edge = split_edge['train']['target_node'].to(data.x.device)

    total_loss = total_examples = 0
    for i, perm in enumerate(DataLoader(range(source_edge.size(0)), batch_size, shuffle=True)):
        # 量化权重
        train_dec.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        if args.call_neurosim:
            train_dec.bind_hooks(model, i, cur_epoch)
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

        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        # 量化梯度
        train_dec.quantify_activation(model)

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
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-citation2',
                                     transform=T.ToSparseTensor())
    data = dataset[0]
    data.adj_t = data.adj_t.to_symmetric()
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

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

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
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder, adj_activity=activity).to(device)

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

    evaluator = Evaluator(name='ogbl-citation2')
    logger = Logger(args.runs, args)

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
            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])

        print('GraphSAGE' if args.use_sage else 'GCN')
        logger.print_statistics(run)
    print('GraphSAGE' if args.use_sage else 'GCN')
    logger.print_statistics()


if __name__ == "__main__":
    main()
