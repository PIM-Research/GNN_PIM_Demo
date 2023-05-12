import argparse
from subprocess import call

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import time
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from models import GCN
from util.global_variable import args, weight_quantification, grad_clip, run_recorder, grad_quantiication
from util.hook import set_vertex_map, set_updated_vertex_map
from util.logger import Logger
from util.other import norm_adj, quantify_adj, store_updated_list_and_adj_matrix, transform_adj_matrix, \
    record_net_structure, get_updated_num
from util.train_decorator import TrainDecorator


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


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


def train(model, predictor, data, split_edge, optimizer, batch_size, train_decorator: TrainDecorator, cur_epoch=0,
          cluster_label=None):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(data.x.device)

    total_loss = total_examples = 0
    dst_vertex_num = 0
    num_i = 0
    for i, perm in enumerate(DataLoader(range(pos_train_edge.size(0)), batch_size,
                                        shuffle=True)):
        start_time = time.perf_counter()
        # 量化权重
        if args.bl_weight != -1:
            train_decorator.quantify_weight(model, i, cur_epoch)

        # 绑定钩子函数，记录各层的输入
        if args.call_neurosim:
            train_decorator.bind_hooks(model, i, cur_epoch)
        optimizer.zero_grad()
        edge = pos_train_edge[perm].t()
        dst_vertex_num += get_updated_num(torch.unique(edge[1]))
        num_i += 1

        if args.filter_adj:
            data.adj_t = train_decorator.filter_adj_by_batch(adj_t=data.adj_t, source_vertexes=edge[0],
                                                             dst_vertexes=edge[1], batch_index=i)

        h = model(data.x, data.adj_t)

        if args.use_cluster:
            src = cluster_label[edge[0]]
            dst = cluster_label[edge[1]]
        else:
            src = edge[0]
            dst = edge[1]
        pos_out = predictor(h[src], h[dst])
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
        if args.bl_weight != -1:
            train_decorator.quantify_grad(model)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        # 清楚钩子
        if args.call_neurosim:
            train_decorator.clear_hooks(model, i, cur_epoch)
        end_time = time.perf_counter()
        print(f'current epoch:{cur_epoch}  current Iteration:{i} epoch time:{start_time-end_time}')
    print('dst_vertex_num_avg:', dst_vertex_num / num_i * (1 - 0.01 * args.percentile))
    print('num_i:', num_i)

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
    writer = SummaryWriter()
    train_dec = TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ppa',
                                     transform=T.ToSparseTensor())

    data = dataset[0]
    data.x = data.x.to(torch.float)
    if args.use_node_embedding:
        data.x = torch.cat([data.x, torch.load('embedding.pt')], dim=-1)

    # 将邻接矩阵正则化
    if args.call_neurosim:
        adj_matrix = norm_adj(data.adj_t)
        if args.bl_activate != -1:
            adj_matrix = quantify_adj(adj_matrix, args.bl_activate)
    else:
        adj_matrix = None

    # 获取词嵌入数量
    cluster_label = None
    if args.use_cluster:
        cluster_label, data.num_nodes, adj_matrix, data.adj_t = transform_adj_matrix(data, device)

    # 获取顶点特征更新列表
    updated_vertex, vertex_pointer = store_updated_list_and_adj_matrix(adj_t=data.adj_t, adj_binary=adj_matrix)
    if vertex_pointer is not None:
        set_vertex_map(vertex_pointer)
    set_updated_vertex_map(updated_vertex)

    if args.call_neurosim:
        record_net_structure(data.num_nodes, data.num_features, args.hidden_channels, args.hidden_channels,
                             args.num_layers)

    split_edge = dataset.get_edge_split()

    data = data.to(device)
    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels, args.hidden_channels, args.num_layers,
                    args.dropout, bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ppa')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }
    if args.percentile != 0:
        train_dec.bind_update_hook(model)
    test_time = 0
    epoch_time = 0

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            start_time = time.perf_counter()
            loss = train(model, predictor, data, split_edge, optimizer, args.batch_size, train_decorator=train_dec,
                         cur_epoch=epoch, cluster_label=cluster_label)
            writer.add_scalar('ppa/Loss', loss, epoch)

            if epoch % args.eval_steps == 0:
                test_s = time.perf_counter()
                results = test(model, predictor, data, split_edge, evaluator,
                               args.batch_size, cluster_label=cluster_label)
                test_e = time.perf_counter()
                test_time += test_e - test_s
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
            end_time = time.perf_counter()
            epoch_time += start_time - end_time
            print('epoch运行时长：', epoch_time - test_time, '秒')
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run, key=key)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics(key=key)

    print('运行时长：', epoch_time - test_time, '秒')


if __name__ == "__main__":
    main()
