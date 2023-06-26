import argparse
import time
from subprocess import call

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from tensorboardX import SummaryWriter
from torch_geometric.nn import SAGEConv

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from util.global_variable import args, weight_quantification, grad_quantiication, grad_clip, run_recorder
from util.hook import set_updated_vertex_map, set_vertex_map
from util.logger import Logger
from models.gcn import GCN
from util.other import norm_adj, quantify_adj, record_net_structure, get_updated_num, store_updated_list_and_adj_matrix
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


def train(model, data, train_idx, optimizer, train_decorator: TrainDecorator, cur_epoch=0):
    if args.call_neurosim:
        train_decorator.create_bash_command(cur_epoch, model.bits_W, model.bits_A)
    model.train()
    # 量化权重
    if args.bl_weight != -1:
        train_decorator.quantify_weight(model, 0, cur_epoch)

    # 绑定钩子函数，记录各层的输入
    if args.call_neurosim:
        train_decorator.bind_hooks(model, 0, cur_epoch)

    if args.filter_adj:
        data.adj_t = train_decorator.filter_adj_by_batch(adj_t=data.adj_t, source_vertexes=train_idx,
                                                         dst_vertexes=train_idx, batch_index=0)
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    # 量化梯度
    if args.bl_grad != -1:
        train_decorator.quantify_grad(model)

    optimizer.step()
    # 清除钩子
    if args.call_neurosim:
        train_decorator.clear_hooks(model, 0, cur_epoch)

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc


def main():
    writer = SummaryWriter()
    train_dec = TrainDecorator(weight_quantification, grad_quantiication, grad_clip, run_recorder)
    # runs为10，epochs为1000，神经网络层数为3层，dropout为0，lr为0.01，隐藏层数为256，batch_size为64*1024
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]

    # Move edge features to node features.
    data.x = data.adj_t.mean(dim=1)
    data.adj_t.set_value_(None)

    # 将邻接矩阵正则化
    if args.call_neurosim:
        adj_matrix = norm_adj(data.adj_t)
        if args.bl_activate != -1:
            adj_matrix = quantify_adj(adj_matrix, args.bl_activate)
    else:
        adj_matrix = None

    # 获取顶点特征更新列表
    updated_vertex, vertex_pointer = store_updated_list_and_adj_matrix(adj_t=data.adj_t, adj_binary=adj_matrix)

    if vertex_pointer is not None:
        set_vertex_map(vertex_pointer)
    set_updated_vertex_map(updated_vertex)

    if args.call_neurosim:
        record_net_structure(data.num_nodes, data.num_features, args.hidden_channels, dataset.num_classes,
                             args.num_layers)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    print('参与Aggregation顶点数：', get_updated_num(train_idx) * (1 - 0.01 * args.percentile))

    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels, 112,
                     args.num_layers, args.dropout).to(device)
    else:
        model = GCN(data.num_features, args.hidden_channels, 112, args.num_layers, args.dropout,
                    bl_weight=args.bl_weight, bl_activate=args.bl_activate, bl_error=args.bl_error,
                    recorder=run_recorder, normalize=False).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)
    if args.percentile != 0:
        train_dec.bind_update_hook(model)
    test_time = 0
    start_time = time.perf_counter()

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)
            if epoch % args.eval_steps == 0:
                writer.add_scalar('arxiv/Loss', loss, epoch)
                test_s = time.perf_counter()
                result = test(model, data, split_idx, evaluator)
                test_e = time.perf_counter()
                test_time += test_e - test_s
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')
                    writer.add_scalar(f'proteins/Train accuracy', 100 * train_rocauc, epoch)
                    writer.add_scalar(f'proteins/Valid accuracy', 100 * valid_rocauc, epoch)
                    writer.add_scalar(f'proteins/Test accuracy', 100 * test_rocauc, epoch)
            if args.call_neurosim:
                call(["chmod", "o+x", run_recorder.bootstrap_path])
                call(["/bin/bash", run_recorder.bootstrap_path])

            if args.use_pipeline:
                vertex_num = data.num_nodes
                input_channels = data.num_features
                hidden_channels = args.hidden_channels
                output_channels = 112

                with open('./pipeline/matrix_info.csv', 'a') as file:
                    file.write(
                        f'{vertex_num},{input_channels},{input_channels},{hidden_channels},'
                        f'{vertex_num},{vertex_num},{vertex_num},'
                        f'{hidden_channels},{epoch},{1}\n')
                    file.write(
                        f'{vertex_num},{hidden_channels},{hidden_channels},{hidden_channels},'
                        f'{vertex_num},{vertex_num},{vertex_num},'
                        f'{hidden_channels},{epoch},{2}\n')
                    file.write(
                        f'{vertex_num},{hidden_channels},{hidden_channels},{output_channels},'
                        f'{vertex_num},{vertex_num},{vertex_num},'
                        f'{output_channels},{epoch},{3}\n')

        logger.print_statistics(run)
    logger.print_statistics()
    end_time = time.perf_counter()
    print('运行时长：', end_time - start_time - test_time, '秒')


if __name__ == "__main__":
    main()
